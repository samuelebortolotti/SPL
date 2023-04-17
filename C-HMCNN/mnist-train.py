import os
import datetime
import json
from time import perf_counter

import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from sklearn.metrics import (
    precision_score,
    average_precision_score,
    hamming_loss,
    jaccard_score
)
from sklearn.model_selection import train_test_split
import wandb

# Circuit imports
import sys
sys.path.append(os.path.join(sys.path[0],'hmc-utils'))
sys.path.append(os.path.join(sys.path[0],'hmc-utils', 'pypsdd'))
sys.path.append(os.path.join(sys.path[0],'dataset'))

from GatingFunction import DenseGatingFunction
from compute_mpe import CircuitMPE
from pysdd.sdd import SddManager, Vtree

# misc
from common import *
from dataset import load_dataloaders, LoadDebugDataset

class RRRLossWithGate(nn.Module):
    def __init__(
        self,
        net: nn.Module,
        gate: DenseGatingFunction,
        cmpe: CircuitMPE,
        regularizer_rate=100,
        weight=None,
        rr_clipping=None,
    ) -> None:
        super().__init__()
        self.gate = gate
        self.cmpe = cmpe
        self.regularizer_rate = regularizer_rate
        self.net = net
        self.weight = weight
        self.rr_clipping = rr_clipping

    def forward(
        self,
        thetas,
        X,
        y,
        expl,
        logits,
        confounded,
        to_eval,
    ):
        # use the basic criterion
        logits = logits[:, to_eval]
        # the normal training loss
        self.cmpe.set_params(thetas)
        right_answer_loss = self.cmpe.cross_entropy(y, log_space=True).mean()
        # change the part of the evaluation
        y = y[:, to_eval]

        # get gradients w.r.t. to the input
        log_prob_ys = F.log_softmax(logits, dim=1)
        log_prob_ys.retain_grad()

        # integrated gradients
        gradXes = None

        # if the example is not confunded from the beginning,
        # then I can simply avoid computing the right reason loss!
        if ((confounded.byte() == 1).sum()).item():
            gradXes = torch.autograd.grad(
                log_prob_ys,
                X,
                torch.ones_like(log_prob_ys),
                create_graph=True,
                allow_unused=True,
            )[0]
        else:
            gradXes = torch.zeros_like(X)

        expl = expl.unsqueeze(dim=1)
        A_gradX = torch.mul(expl, gradXes) ** 2

        # sum each axes contribution
        right_reason_loss = torch.sum(A_gradX)
        right_reason_loss *= self.regularizer_rate

        if self.weight is not None:
            right_reason_loss *= self.weight[y[0]]

        if self.rr_clipping is not None:
            if right_reason_loss > self.rr_clipping:
                right_reason_loss = (
                    right_reason_loss - right_reason_loss + self.rr_clipping
                )

        res = right_answer_loss + right_reason_loss

        return (
            res,
            right_answer_loss,
            right_reason_loss,
        )


def revise_step_with_gates(
    net: nn.Module,
    gate: DenseGatingFunction,
    cmpe: CircuitMPE,
    debug_loader: torch.utils.data.DataLoader,
    train,
    R: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    revive_function: RRRLossWithGate,
    epoch,
    num_epochs,
    device: str = "cuda",
    have_to_train: bool = True,
    title=""
):
    comulative_loss = 0.0
    cumulative_accuracy = 0.0
    cumulative_right_answer_loss = 0.0
    cumulative_right_reason_loss = 0.0
    confounded_samples = 0.0

    # set the network to training mode
    if have_to_train:
        net.train()
        gate.train()
    else:
        net.eval()
        gate.eval()

    for batch_idx, inputs in enumerate(debug_loader):
        (sample, ground_truth, confounder_mask, confounded, superc, subc) = inputs

        # load data into device
        sample = sample.to(device)
        sample.requires_grad = True
        # ground_truth element
        ground_truth = ground_truth.to(device)
        # confounder mask
        confounder_mask = confounder_mask.to(device)
        # confounded
        confounded = confounded.to(device)

        # gradients reset
        if have_to_train:
            optimizer.zero_grad()

        # output
        outputs = net(sample.float())
        thetas = gate(outputs.float())
        cmpe.set_params(thetas)
        predicted = (cmpe.get_mpe_inst(sample.shape[0]) > 0).long()

        # general prediction loss computation
        # MCLoss (their loss)
        constr_output = get_constr_out(outputs, R)
        train_output = ground_truth * outputs.double()
        train_output = get_constr_out(train_output, R)
        train_output = (
            1 - ground_truth
        ) * constr_output.double() + ground_truth * train_output

        # get the loss masking the prediction on the root -> confunder
        (loss, right_answer_loss, right_reason_loss,) = revive_function(
            thetas=thetas,
            X=sample,
            y=ground_truth,
            expl=confounder_mask,
            logits=train_output,
            confounded=confounded,
            to_eval=train.to_eval,
        )

        # compute the amount of confounded samples
        confounded_samples += confounded.sum().item()

        # compute training accuracy
        cumulative_accuracy += (predicted == ground_truth.byte()).sum().item()

        # for calculating loss, acc per epoch
        comulative_loss += loss.item()
        cumulative_right_answer_loss += right_answer_loss.item()
        cumulative_right_reason_loss += right_reason_loss.item()

        if have_to_train:
            # backward pass
            loss.backward()
            # optimizer
            optimizer.step()

        # compute the au(prc)
        predicted = predicted.to("cpu")
        ground_truth = ground_truth.to("cpu")

        if batch_idx == 0:
            predicted_train = predicted
            y_test = ground_truth
            output_train = train_output.detach().clone()
        else:
            predicted_train = torch.cat((predicted_train, predicted), dim=0)
            output_train = torch.cat(
                (output_train, train_output.detach().clone()), dim=0
            )
            y_test = torch.cat((y_test, ground_truth), dim=0)

        # TODO force exit
        if batch_idx == 200:
            break

    y_test = y_test[:, train.to_eval]
    predicted_train = predicted_train.data[:, train.to_eval].to(torch.float)
    output_train = output_train[:, train.to_eval]

    # jaccard score
    jaccard = jaccard_score(y_test, predicted_train, average="micro")
    # hamming score
    hamming = hamming_loss(y_test, predicted_train)
    # average precision score
    auprc_score = average_precision_score(y_test, output_train, average="micro")
    # accuracy
    accuracy = cumulative_accuracy / len(y_test)

    # confounded samples
    if confounded_samples == 0:
        confounded_samples = 1

    print(
        "\n\t {}: [Epoch {}/{}] loss {:.5f}, accuracy {:.2f}%, Jaccard Score {:.3f}, Hamming Score {:.3f}, Area under Precision-Recall Curve Raw {:.3f}".format(
            title, epoch+1, num_epochs, comulative_loss / len(debug_loader), accuracy, jaccard, hamming, auprc_score
        )
    )

    return (
        comulative_loss / len(debug_loader),
        cumulative_right_answer_loss / len(debug_loader),
        cumulative_right_reason_loss / len(debug_loader),
        cumulative_right_reason_loss / confounded_samples,
        accuracy,
        auprc_score,
        hamming,
        jaccard,
    )

def get_lr(optimizer: torch.optim.Optimizer):
    r"""
    Function which returns the learning rate value
    used in the optimizer

    Args:
        optimizer [torch.optim.Optimizer]: optimizer

    Returns:
        lr [float]: learning rate
    """
    for param_group in optimizer.param_groups:
        return param_group["lr"]

def log1mexp(x):
        assert(torch.all(x >= 0))
        return torch.where(x < 0.6931471805599453094, torch.log(-torch.expm1(-x)), torch.log1p(-torch.exp(-x)))

class ConstrainedFFNNModel(nn.Module):
    """ C-HMCNN(h) model - during training it returns the not-constrained output that is then passed to MCLoss """
    def __init__(self, input_dim, hidden_dim, output_dim, hyperparams, R):
        super(ConstrainedFFNNModel, self).__init__()

        self.nb_layers = hyperparams['num_layers']
        self.R = R

        fc = []
        for i in range(self.nb_layers):
            if i == 0:
                fc.append(nn.Linear(input_dim, hidden_dim))
            elif i == self.nb_layers-1:
                fc.append(nn.Linear(hidden_dim, output_dim))
            else:
                fc.append(nn.Linear(hidden_dim, hidden_dim))
        self.fc = nn.ModuleList(fc)

        self.drop = nn.Dropout(hyperparams['dropout'])

        if hyperparams['non_lin'] == 'tanh':
            self.f = nn.Tanh()
        else:
            self.f = nn.ReLU()

    def forward(self, x, sigmoid=False, log_sigmoid=False):
        x = x.view(-1, 28 * 28 * 1)
        for i in range(self.nb_layers):
            if i == self.nb_layers-1:
                if sigmoid:
                    x = nn.Sigmoid()(self.fc[i](x))
                elif log_sigmoid:
                    x = nn.LogSigmoid()(self.fc[i](x))
                else:
                    x = self.fc[i](x)
            else:
                x = self.f(self.fc[i](x))
                x = self.drop(x)

        if self.R is None:
            return x

        if self.training:
            constrained_out = x
        else:
            constrained_out = get_constr_out(x, self.R)
        return constrained_out

def main():

    args = parse_args()
    print("Args", args)

    set_wandb = False
    # set wandb if needed
    if args.wandb:
        # set the argument to true
        set_wandb = args.wandb
        # Log in to your W&B account
        wandb.login()
        wandb.init(project=args.project, entity=args.entity)

    num_epochs = args.n_epochs
    dataset_name = "mnist"
    hidden_dim = 200

    # Set the hyperparameters 
    hyperparams = {
        'num_layers': 3,
        'dropout': 0.7,
        'non_lin': 'relu',
    }

    # Set seed
    seed_all_rngs(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda:" + str(args.device) if torch.cuda.is_available() else "cpu")

    dataloaders = load_dataloaders(
        img_size=28, img_depth=1, device=args.device
    )

    train = dataloaders['train']
    test = dataloaders['test']
    A = dataloaders['train_set'].get_A()
    confounder_mask_train = LoadDebugDataset(
        dataloaders["train_dataset_with_labels_and_confunders_position"],
    )
    confounder_mask_test = LoadDebugDataset(
        dataloaders["test_dataset_with_labels_and_confunders_pos"],
    )

    train_loader = dataloaders['train_loader']
    valid_loader = dataloaders['val_loader']
    test_loader = dataloaders['test_loader']
    test_debug = torch.utils.data.DataLoader(
        confounder_mask_test,
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=4,
    )
    debug_loader = torch.utils.data.DataLoader(
        confounder_mask_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
    )

    print("Dataloader prepared")

    if not args.no_constraints or False:

        if not os.path.isfile('constraints/' + dataset_name + '.sdd') or not os.path.isfile('constraints/' + dataset_name + '.vtree'):
            # Compute matrix of ancestors R
            # Given n classes, R is an (n x n) matrix where R_ij = 1 if class i is ancestor of class j
            R = np.zeros(A.shape)
            np.fill_diagonal(R, 1)
            g = nx.DiGraph(A)
            for i in range(len(A)):
                descendants = list(nx.descendants(g, i))
                if descendants:
                    R[i, descendants] = 1
            R = torch.tensor(R)

            #Transpose to get the ancestors for each node 
            R = R.unsqueeze(0).to(device)

            # Uncomment below to compile the constraint
            R.squeeze_()
            mgr = SddManager(
                var_count=R.size(0),
                auto_gc_and_minimize=True)

            alpha = mgr.true()
            alpha.ref()
            for i in range(R.size(0)):

               beta = mgr.true()
               beta.ref()
               for j in range(R.size(0)):

                   if R[i][j] and i != j:
                       old_beta = beta
                       beta = beta & mgr.vars[j+1]
                       beta.ref()
                       old_beta.deref()

               old_beta = beta
               beta = -mgr.vars[i+1] | beta
               beta.ref()
               old_beta.deref()

               old_alpha = alpha
               alpha = alpha & beta
               alpha.ref()
               old_alpha.deref()

            # Saving circuit & vtree to disk
            alpha.save(str.encode('constraints/' + dataset_name + '.sdd'))
            alpha.vtree().save(str.encode('constraints/' + dataset_name + '.vtree'))

        # Create circuit object
        cmpe = CircuitMPE('constraints/' + dataset_name + '.vtree', 'constraints/' + dataset_name + '.sdd')

        if args.S > 0:
            cmpe.overparameterize(S=args.S)
            print("Done overparameterizing")

        # Create gating function
        gate = DenseGatingFunction(cmpe.beta, gate_layers=[67] + [67]*args.gates, num_reps=args.num_reps).to(device)
        R = None

        print("CMPE e DenseGating function prepared...")

    else:
        # Use fully-factorized sdd
        mgr = SddManager(var_count=train.A.shape[0], auto_gc_and_minimize=True)
        alpha = mgr.true()
        vtree = Vtree(var_count = train.A.shape[0], var_order=list(range(1, train.A.shape[0] + 1)))
        alpha.save(str.encode('ancestry.sdd'))
        vtree.save(str.encode('ancestry.vtree'))
        cmpe = CircuitMPE('ancestry.vtree', 'ancestry.sdd')
        cmpe.overparameterize()

        # Gating function
        gate = DenseGatingFunction(cmpe.beta, gate_layers=[67]).to(device)
        R = None

    # Output path
    if args.exp_id:
        out_path = os.path.join(args.output, args.exp_id)
    else:
        date_string = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        out_path = os.path.join(args.output,  '{}_{}_{}_{}_{}'.format("mnist", date_string, args.batch_size, args.gates, args.lr))
    os.makedirs(out_path, exist_ok=True)

    # Tensorboard
    #  writer = SummaryWriter(log_dir=os.path.join(out_path, "runs"))

    # Dump experiment parameters
    args_out_path = os.path.join(out_path, 'args.json')
    json_args = json.dumps(vars(args))

    print("Starting with arguments:\n%s\n\tdumped at %s", json_args, args_out_path)
    with open(args_out_path, 'w') as f:
        f.write(json_args)

    # Create the model
    model = ConstrainedFFNNModel(28*28*1, hidden_dim, 67, hyperparams, R)
    model.to(device)

    if set_wandb:
        wandb.watch(model)

    print("Model on gpu", next(model.parameters()).is_cuda)
    optimizer = torch.optim.Adam(list(model.parameters()) + list(gate.parameters()), lr=args.lr, weight_decay=args.wd)

    def evaluate_circuit(model, gate, cmpe, epoch, data_loader, data_split, prefix, title):

        test_correct = 0
        tot_accuracy = 0
        tot_loss = 0
        test_val_t = perf_counter()

        for i, (x,y) in enumerate(data_loader):

            model.eval()
            gate.eval()

            x = x.to(device)
            y = y.to(device)

            # Parameterize circuit using nn
            emb = model(x.float())
            thetas = gate(emb)

            # negative log likelihood and map
            cmpe.set_params(thetas)
            nll = cmpe.cross_entropy(y, log_space=True).mean()

            cmpe.set_params(thetas)
            pred_y = (cmpe.get_mpe_inst(x.shape[0]) > 0).long()

            cmpe.set_params(thetas)
            loss = cmpe.cross_entropy(y, log_space=True).mean()

            pred_y = pred_y.to('cpu')
            y = y.to('cpu')

            num_correct = (pred_y == y.byte()).all(dim=-1).sum()
            tot_loss += loss

            if i == 0:
                test_correct = num_correct
                output_train = emb
                predicted_test = pred_y
                y_test = y
            else:
                test_correct += num_correct
                predicted_test = torch.cat((predicted_test, pred_y), dim=0)
                output_train = torch.cat((output_train, emb), dim=0)
                y_test = torch.cat((y_test, y), dim=0)

            # TODO increase
            if i == 200:
                break

        dt = perf_counter() - test_val_t
        y_test = y_test[:,data_split.to_eval]
        predicted_test = predicted_test[:,data_split.to_eval]
        output_train = output_train.data[:, data_split.to_eval]

        # jaccard score
        jaccard = jaccard_score(y_test, predicted_test, average="micro")
        # hamming score
        hamming = hamming_loss(y_test, predicted_test)
        # average precision score
        auprc_score = average_precision_score(y_test, output_train, average="micro")
        # accuracy
        accuracy = tot_accuracy / len(y_test)
        # nll
        nll = nll.detach().to("cpu").numpy() / (i+1)

        print(
            "\n\t {}: [Epoch {}/{}] loss {:.5f}, accuracy {:.2f}%, Jaccard Score {:.3f}, Hamming Score {:.3f}, Area under Precision-Recall Curve Raw {:.3f}, Time {:.4f}".format(
                title, epoch+1, num_epochs, tot_loss / len(train_loader), accuracy, jaccard, hamming, auprc_score, dt
            )
        )

        return tot_loss / len(train_loader), accuracy, jaccard, hamming, auprc_score


    print("Running epochs...")
    for epoch in range(num_epochs):

        print(f"EVAL@{epoch}")

        (
            revise_total_loss,
            revise_total_right_answer_loss,
            revise_total_right_reason_loss,
            revise_right_reason_loss_confounded,
            revise_total_accuracy,
            revise_total_score_raw,
            revise_hamming_score,
            revise_jaccard_score,
        ) = revise_step_with_gates(
            net=model,
            gate=gate,
            cmpe=cmpe,
            epoch=epoch,
            num_epochs=num_epochs,
            debug_loader=iter(debug_loader),
            R=dataloaders["train_R"],
            train=dataloaders["train"],
            optimizer=optimizer,
            revive_function=RRRLossWithGate(
                net=model,
                gate=gate,
                cmpe=cmpe,
                regularizer_rate=1,
            ),
            device=args.device,
            have_to_train=False,
            title="Test Revive on Train"
        )

        (
            test_revise_total_loss,
            test_revise_total_right_answer_loss,
            test_revise_total_right_reason_loss,
            test_revise_right_reason_loss_confounded,
            test_revise_total_accuracy,
            test_revise_total_score_raw,
            test_revise_hamming_score,
            test_revise_jaccard_score,
        ) = revise_step_with_gates(
            net=model,
            gate=gate,
            cmpe=cmpe,
            epoch=epoch,
            num_epochs=num_epochs,
            debug_loader=iter(test_debug),
            R=dataloaders["train_R"],
            train=dataloaders["train"],
            optimizer=optimizer,
            revive_function=RRRLossWithGate(
                net=model,
                regularizer_rate=1,
                gate=gate,
                cmpe=cmpe,
            ),
            device=args.device,
            have_to_train=False,
            title="Test Revive on Train"
        )

        (test_loss, test_accuracy, test_jaccard, test_hamming, test_auprc) = evaluate_circuit(
            model,
            gate,
            cmpe,
            epoch=epoch,
            data_loader=test_loader,
            data_split=test,
            prefix="param_sdd/test",
            title="Test"
        )
        (val_loss, val_accuracy, val_jaccard, val_hamming, val_auprc) = evaluate_circuit(
            model,
            gate,
            cmpe,
            epoch=epoch,
            data_loader=valid_loader,
            data_split=train,
            prefix="param_sdd/valid",
            title="Validation"
        )

        train_t = perf_counter()

        model.train()
        gate.train()

        tot_loss = 0
        tot_accuracy = 0
        for i, (x, labels) in enumerate(train_loader):

            x = x.to(device)
            labels = labels.to(device)

            # Clear gradients w.r.t. parameters
            optimizer.zero_grad()

            #MCLoss
            if args.no_constraints:
                # Use fully-factorized distribution via circuit
                output = model(x.float(), sigmoid=False)
                thetas = gate(output)
                cmpe.set_params(thetas)
                loss = cmpe.cross_entropy(labels, log_space=True).mean()

                # predicted
                cmpe.set_params(thetas)
                predicted = (cmpe.get_mpe_inst(x.shape[0]) > 0).long()
            else:
                output = model(x.float(), sigmoid=False)
                thetas = gate(output)
                cmpe.set_params(thetas)
                loss = cmpe.cross_entropy(labels, log_space=True).mean()

                # predicted
                cmpe.set_params(thetas)
                predicted = (cmpe.get_mpe_inst(x.shape[0]) > 0).long()

            tot_accuracy += (predicted == labels.byte()).all(dim=-1).sum()
            tot_loss += loss
            loss.backward()
            optimizer.step()

            if i == 0:
                predicted_train = predicted
                output_train = output
                y_test = labels
            else:
                predicted_train = torch.cat((predicted_train, predicted), dim=0)
                output_train = torch.cat((output_train, output), dim=0)
                y_test = torch.cat((y_test, labels), dim=0)

            # TODO increase
            if i == 200:
                break

        y_test = y_test[:, train.to_eval]
        predicted_train = predicted_train.data[:, train.to_eval].to(torch.float)
        output_train = output_train.data[:, train.to_eval].to(torch.float)

        # jaccard score
        jaccard = jaccard_score(y_test, predicted_train, average="micro")
        # hamming score
        hamming = hamming_loss(y_test, predicted_train)
        # average precision score
        auprc_score = average_precision_score(y_test, output_train, average="micro")
        # accuracy
        accuracy = tot_accuracy / len(y_test)

        train_e = perf_counter()
        print(
            "\n\t Train: [Epoch {}/{}] loss {:.5f}, accuracy {:.2f}%, Jaccard Score {:.3f}, Hamming Score {:.3f}, Area under Precision-Recall Curve Raw {:.3f} Time: {:.4f}".format(
                epoch+1, num_epochs, tot_loss / len(train_loader), accuracy, jaccard, hamming, auprc_score, train_e-train_t
            )
        )

        logs = {
            "train/train_loss": tot_loss / len(train_loader),
            "train/train_accuracy": accuracy,
            "train/train_jaccard": jaccard,
            "train/train_hamming": hamming,
            "train/train_auprc_raw": auprc_score,
            "train/train_right_anwer_loss": tot_loss / len(train_loader),
            "train/train_right_reason_loss": revise_total_right_reason_loss,
            "val/val_loss": val_loss,
            "val/val_accuracy": val_accuracy,
            "val/val_jaccard": val_jaccard,
            "val/val_hamming": val_hamming,
            "val/val_auprc_raw": val_auprc,
            "test/test_loss": test_loss,
            "test/test_accuracy": test_accuracy,
            "test/test_jaccard": test_jaccard,
            "test/test_hamming": test_hamming,
            "test/test_auprc_raw": test_auprc,
            "test/test_right_answer_loss": test_loss,
            "test/test_right_reason_loss": test_revise_total_right_reason_loss,
            "learning_rate": get_lr(optimizer),
        }

        if set_wandb:
            wandb.log(logs)

if __name__ == "__main__":
    print("In main...")
    main()
