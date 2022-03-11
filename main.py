import argparse, time, os, random, math
import pprint as pp
from utils_data import *
from utils_algo import *
from utils_mcl_loss import *
from models import *
from attack_generator import *
from utils_func import *


def one_stage_adversarial_train(args, model, optimizer, seed):
    """
        one-stage adversarial training method,
        only for single cl setting under uniform assumption.
    """
    print(">> One-stage Adversarial learning with Complementary Labels")
    loss_fn, lr, best_nat_acc, best_pgd20_acc, best_cw_acc, best_epoch = create_loss_fn(args), args.at_lr, 0, 0, 0, 0
    nature_test_acc_list, pgd20_acc_list, cw_acc_list = [], [], []
    first_layer_grad, last_layer_grad = [], []
    for epoch in range(args.adv_epochs):
        lr = adv_lr_schedule(lr, epoch + 1, optimizer)
        for batch_idx, (images, cl_labels, true_labels, id) in enumerate(complementary_train_loader):
            images, cl_labels, true_labels = images.to(device), cl_labels.to(device), true_labels.to(device)
            epsilon, step_size, num_steps = at_param_schedule(args, epoch + 1)

            # x_adv = pgd(model, images, true_labels, epsilon, step_size, num_steps, loss_fn="cent", category="Madry", rand_init=True, num_classes=K)  # mex_ce_tl
            # x_adv = pgd(model, images, None, epsilon, step_size, num_steps, loss_fn="kl", category="trades", rand_init=True, num_classes=K)  # max_trades
            x_adv = cl_adv(args, model, images, cl_labels, epsilon, step_size, num_steps, id, ccp, partialY, loss_fn, category="Madry", rand_init=True)  # max_cl

            model.train()
            optimizer.zero_grad()
            logit = model(x_adv)
            if args.method in ['exp', 'log']:
                loss = loss_fn(logit, partialY[id].float())
            elif args.method in ['mae', 'mse', 'ce', 'gce', 'phuber_ce']:
                loss = unbiased_estimator(loss_fn, logit, partialY[id].float())
            elif args.method in ['free', 'nn', 'ga', 'pc', 'forward', 'scl_exp', 'scl_nl', 'l_uw', 'l_w']:
                assert args.cl_num == 1
                loss, _ = chosen_loss_c(f=logit, K=K, labels=cl_labels, ccp=ccp, meta_method=args.method)
            loss.backward()
            # the grad stat
            assert args.model == "cnn", "please modify grad stat code!"
            first_layer_grad.append(model.module.feature_extractor.conv1.weight.grad.norm(p=2).cpu().item())
            last_layer_grad.append(model.module.classifier.fc3.weight.grad.norm(p=2).cpu().item())
            optimizer.step()

        # Evalutions
        model.eval()
        test_nat_acc = accuracy_check(loader=test_loader, model=model)
        _, test_pgd20_acc = eval_robust(model, test_loader, perturb_steps=20, epsilon=args.epsilon, step_size=args.step_size, loss_fn="cent", category="Madry", random=True, num_classes=K)
        _, test_cw_acc = eval_robust(model, test_loader, perturb_steps=30, epsilon=args.epsilon, step_size=args.step_size, loss_fn="cw", category="Madry", random=True, num_classes=K)
        nature_test_acc_list.append(test_nat_acc)
        pgd20_acc_list.append(test_pgd20_acc)
        cw_acc_list.append(test_cw_acc)
        print(epsilon, step_size, num_steps)
        print('Epoch: [%d | %d] | Learning Rate: %f | Natural Test Acc %.4f | PGD20 Test Acc %.4f | CW Test Acc %.4f |\n' % (epoch + 1, args.adv_epochs, lr, test_nat_acc, test_pgd20_acc, test_cw_acc))

        # Save the best checkpoint
        if test_pgd20_acc > best_pgd20_acc:
            best_epoch = epoch + 1
            best_nat_acc = test_nat_acc
            best_pgd20_acc = test_pgd20_acc
            best_cw_acc = test_cw_acc
            torch.save({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'test_nat_acc': test_nat_acc,
                'test_pgd20_acc': test_pgd20_acc,
                'test_cw_acc': test_cw_acc,
                'optimizer': optimizer.state_dict(),
            }, os.path.join(args.out_dir, "best_checkpoint_seed{}.pth.tar".format(seed)))

        # Save the last checkpoint
        torch.save({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'test_nat_acc': test_nat_acc,
            'test_pgd20_acc': test_pgd20_acc,
            'test_cw_acc': test_cw_acc,
            'optimizer': optimizer.state_dict(),
        }, os.path.join(args.out_dir, "checkpoint_seed{}.pth.tar".format(seed)))

    print(nature_test_acc_list)
    print(pgd20_acc_list)
    print(cw_acc_list)
    print(">> Finished Adv Training: Natural Test Acc | Last_checkpoint %.4f | Best_checkpoint(%.1f) %.4f |\n" % (test_nat_acc, best_epoch, best_nat_acc))
    print(">> Finished Adv Training: PGD20 Test Acc | Last_checkpoint %.4f | Best_checkpoint %.4f |\n" % (test_pgd20_acc, best_pgd20_acc))
    print(">> Finished Adv Training: CW Test Acc | Last_checkpoint %.4f | Best_checkpoint %.4f |\n" % (test_cw_acc, best_cw_acc))
    epoch = [i for i in range(args.adv_epochs)]
    show([epoch, epoch, epoch], [nature_test_acc_list, pgd20_acc_list, cw_acc_list], label=["nature test acc", "pgd20 acc", "cw acc"],
         title=args.dataset, xdes="Epoch", ydes="Test Accuracy", path=os.path.join(args.out_dir, "adv_test_acc_seed{}.png".format(seed)))
    print("first_layer [:2500] iter: \n{} \nfirst_layer [-2500:] iter: \n{} \nlast_layer [:2500] iter: \n{} \nlast_layer [-2500:] iter: \n{}".format(
            first_layer_grad[:2500], first_layer_grad[-2500:], last_layer_grad[:2500], last_layer_grad[-2500:]), file=open(os.path.join(args.out_dir, "grad_seed{}.out".format(seed)), "a+"))
    # Auto-attack
    aa_eval(args, model, filename="aa_last.txt")
    best_checkpoint = torch.load(os.path.join(args.out_dir, "best_checkpoint_seed{}.pth.tar".format(seed)))
    model.load_state_dict(best_checkpoint['state_dict'])
    aa_eval(args, model, filename="aa_best.txt")

    return [test_nat_acc, test_pgd20_acc, test_cw_acc], [best_nat_acc, best_pgd20_acc, best_cw_acc]


def two_stage_adversarial_train(args, model, optimizer, seed):
    """
        two-stage adversarial training method,
        only for single cl setting under uniform assumption.
    """
    print(">> Two-stage Adversarial learning with Complementary Labels")
    lr, best_nat_acc, best_pgd20_acc, best_cw_acc, best_epoch = args.at_lr, 0, 0, 0, 0
    nature_test_acc_list, pgd20_acc_list, cw_acc_list = [], [], []
    cl_model = torch.nn.DataParallel(create_model(args, input_dim, input_channel, K).to(device))
    checkpoint = torch.load(os.path.join(args.out_dir, "cl_best_checkpoint_seed{}.pth.tar".format(seed)))
    cl_model.load_state_dict(checkpoint['state_dict'])
    cl_model.eval()
    print(">> Load the CL model with test acc: {}, epoch {}".format(checkpoint['test_acc'], checkpoint['epoch']))
    for epoch in range(args.adv_epochs):
        lr = adv_lr_schedule(lr, epoch + 1, optimizer)
        for batch_idx, (images, cl_labels, true_labels, id) in enumerate(complementary_train_loader):
            pseudo_labels, prob = get_pred(cl_model, images.detach())
            images, pseudo_labels = images.to(device), pseudo_labels.to(device)
            # 1. for two-stage baseline
            x_adv = pgd(model, images, pseudo_labels, args.epsilon, args.step_size, args.num_steps, loss_fn="cent", category="Madry", rand_init=True, num_classes=K)
            model.train()
            optimizer.zero_grad()
            logit = model(x_adv)
            loss = nn.CrossEntropyLoss(reduction="mean")(logit, pseudo_labels)
            loss.backward()
            optimizer.step()

        # Evalutions
        model.eval()
        test_nat_acc = accuracy_check(loader=test_loader, model=model)
        _, test_pgd20_acc = eval_robust(model, test_loader, perturb_steps=20, epsilon=args.epsilon, step_size=args.step_size, loss_fn="cent", category="Madry", random=True, num_classes=K)
        _, test_cw_acc = eval_robust(model, test_loader, perturb_steps=30, epsilon=args.epsilon, step_size=args.step_size, loss_fn="cw", category="Madry", random=True, num_classes=K)
        nature_test_acc_list.append(test_nat_acc)
        pgd20_acc_list.append(test_pgd20_acc)
        cw_acc_list.append(test_cw_acc)

        print('Epoch: [%d | %d] | Learning Rate: %f | Natural Test Acc %.4f | PGD20 Test Acc %.4f | CW Test Acc %.4f |\n' % (epoch + 1, args.adv_epochs, lr, test_nat_acc, test_pgd20_acc, test_cw_acc))

        # Save the best checkpoint
        if test_pgd20_acc > best_pgd20_acc:
            best_epoch = epoch + 1
            best_nat_acc = test_nat_acc
            best_pgd20_acc = test_pgd20_acc
            best_cw_acc = test_cw_acc
            torch.save({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'test_nat_acc': test_nat_acc,
                'test_pgd20_acc': test_pgd20_acc,
                'test_cw_acc': test_cw_acc,
                'optimizer': optimizer.state_dict(),
            }, os.path.join(args.out_dir, "best_checkpoint_seed{}.pth.tar".format(seed)))

        # Save the last checkpoint
        torch.save({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'test_nat_acc': test_nat_acc,
            'test_pgd20_acc': test_pgd20_acc,
            'test_cw_acc': test_cw_acc,
            'optimizer': optimizer.state_dict(),
        }, os.path.join(args.out_dir, "checkpoint_seed{}.pth.tar".format(seed)))

    print(nature_test_acc_list)
    print(pgd20_acc_list)
    print(cw_acc_list)
    print(">> Finished Adv Training: Natural Test Acc | Last_checkpoint %.4f | Best_checkpoint(%.1f) %.4f |\n" % (test_nat_acc, best_epoch, best_nat_acc))
    print(">> Finished Adv Training: PGD20 Test Acc | Last_checkpoint %.4f | Best_checkpoint %.4f |\n" % (test_pgd20_acc, best_pgd20_acc))
    print(">> Finished Adv Training: CW Test Acc | Last_checkpoint %.4f | Best_checkpoint %.4f |\n" % (test_cw_acc, best_cw_acc))
    epoch = [i for i in range(args.adv_epochs)]
    show([epoch, epoch, epoch], [nature_test_acc_list, pgd20_acc_list, cw_acc_list], label=["nature test acc", "pgd20 acc", "cw acc"],
         title=args.dataset, xdes="Epoch", ydes="Test Accuracy", path=os.path.join(args.out_dir, "adv_test_acc_seed{}.png".format(seed)))
    # Auto-attack
    aa_eval(args, model, filename="aa_last.txt")
    best_checkpoint = torch.load(os.path.join(args.out_dir, "best_checkpoint_seed{}.pth.tar".format(seed)))
    model.load_state_dict(best_checkpoint['state_dict'])
    aa_eval(args, model, filename="aa_best.txt")

    return [test_nat_acc, test_pgd20_acc, test_cw_acc], [best_nat_acc, best_pgd20_acc, best_cw_acc]


def complementary_learning(args, model, optimizer, partialY, seed):
    """
        for learning with single & multiple complementary labels under uniform assumption.
    """
    print(">> Learning with Complementary Labels")
    loss_fn, loss_vector = create_loss_fn(args), None
    lr, best_acc, best_epoch, train_acc_list, test_acc_list = args.cl_lr, 0, 0, [], []
    for epoch in range(args.epochs):
        lr = cl_lr_schedule(lr, epoch + 1, optimizer)
        for i, (images, cl_labels, true_labels, id) in enumerate(complementary_train_loader):
            images, cl_labels = images.to(device), cl_labels.to(device)
            model.train()
            optimizer.zero_grad()
            if args.clat:
                x_adv = cl_adv(args, model, images, cl_labels, args.epsilon, args.step_size, args.num_steps, id, ccp, partialY, loss_fn, category="Madry", rand_init=True)
                optimizer.zero_grad()
                outputs = model(x_adv)
            else:
                outputs = model(images)

            if args.method in ['exp', 'log']:
                loss = loss_fn(outputs, partialY[id].float())
            elif args.method in ['mae', 'mse', 'ce', 'gce', 'phuber_ce']:
                loss = unbiased_estimator(loss_fn, outputs, partialY[id].float())
            elif args.cl_num == 1 and args.method in ['free', 'nn', 'ga', 'pc', 'forward', 'scl_exp', 'scl_nl', 'l_uw', 'l_w']:
                loss, loss_vector = chosen_loss_c(f=outputs, K=K, labels=cl_labels, ccp=ccp, meta_method=args.method)
            if args.method == 'ga':
                if torch.min(loss_vector).item() < 0:
                    loss_vector_with_zeros = torch.cat(
                        (loss_vector.view(-1, 1), torch.zeros(K, requires_grad=True).view(-1, 1).to(device)), 1)
                    min_loss_vector, _ = torch.min(loss_vector_with_zeros, dim=1)
                    loss = torch.sum(min_loss_vector)
                    loss.backward()
                    for group in optimizer.param_groups:
                        for p in group['params']:
                            p.grad = -1 * p.grad
                else:
                    loss.backward()
            else:
                loss.backward()
            optimizer.step()

        model.eval()
        train_accuracy = accuracy_check(loader=train_loader, model=model)
        test_accuracy = accuracy_check(loader=test_loader, model=model)
        train_acc_list.append(train_accuracy)
        test_acc_list.append(test_accuracy)
        print('Epoch: {}. Train Acc: {}. Test Acc: {}.'.format(epoch + 1, train_accuracy, test_accuracy))

        # Save the best checkpoint
        if test_accuracy > best_acc:
            best_epoch = epoch + 1
            best_acc = test_accuracy
            torch.save({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'train_acc': train_accuracy,
                'test_acc': test_accuracy,
                'optimizer': optimizer.state_dict(),
            }, os.path.join(args.out_dir, "cl_best_checkpoint_seed{}.pth.tar".format(seed)))

        # Save the last checkpoint
        torch.save({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'train_acc': train_accuracy,
            'test_acc': test_accuracy,
            'optimizer': optimizer.state_dict(),
        }, os.path.join(args.out_dir, "cl_checkpoint_seed{}.pth.tar".format(seed)))

    # stat
    wrong_count, correct_count = stat(x_to_mcls, x_to_tls)
    print("Epoch {}: {}% data are given wrong complementary labels, and each data are given {} correct complementary labels on average!".format(epoch + 1, wrong_count * 100, correct_count))

    # plot
    print(train_acc_list)
    print(test_acc_list)
    print(">> Best test acc({}): {}".format(best_epoch, max(test_acc_list)))
    print(">> AVG test acc of last 10 epochs: {}".format(np.mean(test_acc_list[-10:])))
    epoch = [i for i in range(args.epochs)]
    show([epoch] * 2, [train_acc_list, test_acc_list], label=["train acc", "test acc"], title=args.dataset, xdes="Epoch", ydes="Accuracy", path=os.path.join(args.out_dir, "cl_acc_seed{}.png".format(seed)))

    return np.mean(test_acc_list[-10:])


def aa_eval(args, model, filename):
    """
        AutoAttack evaluation - pip install git+https://github.com/fra31/auto-attack
    """
    from autoattack import AutoAttack
    model.eval()
    version, norm, individual, n_ex = "standard", "Linf", False, 10000
    adversary = AutoAttack(model, norm=norm, eps=args.epsilon, log_path=os.path.join(args.out_dir, filename), version=version)

    l = [x for (x, y) in test_loader]
    x_test = torch.cat(l, 0)
    l = [y for (x, y) in test_loader]
    y_test = torch.cat(l, 0)

    # run attack and save images
    with torch.no_grad():
        if not individual:
            adv_complete = adversary.run_standard_evaluation(x_test[:n_ex], y_test[:n_ex], bs=500)
            # torch.save({'adv_complete': adv_complete}, '{}/{}_{}_1_{}_eps_{:.5f}.pth'.format(args.out_dir, 'aa', version, adv_complete.shape[0], args.epsilon))
        else:
            # individual version, each attack is run on all test points
            adv_complete = adversary.run_standard_evaluation_individual(x_test[:n_ex], y_test[:n_ex], bs=500)
            # torch.save(adv_complete, '{}/{}_{}_individual_1_{}_eps_{:.5f}_plus_{}_cheap_{}.pth'.format(args.out_dir, 'aa', version, n_ex, args.epsilon))


def at_param_schedule(args, epoch):
    warmup_epoch = args.sch_epoch
    if args.scheduler == "linear":
        eps = min(args.epsilon * (epoch/warmup_epoch), args.epsilon)
    elif args.scheduler == "cosine":
        eps = 1/2 * (1-math.cos(math.pi * min((epoch/warmup_epoch), 1))) * args.epsilon
    elif args.scheduler == "none":
        eps = args.epsilon
    if epoch < warmup_epoch:
        # return eps, eps, 1
        return eps, args.step_size, args.num_steps
    else:
        return args.epsilon, args.step_size, args.num_steps


def cl_lr_schedule(lr, epoch, optimizer):
    if args.dataset == "cifar10" and epoch % 30 == 0:
        lr /= 2
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def adv_lr_schedule(lr, epoch, optimizer):
    if args.dataset != "cifar10":
        pass
    elif args.dataset == "cifar10":
        # 100 epochs for ResNet18/WRN32-10
        if epoch == 50:
            lr /= 10
        if epoch == 75:
            lr /= 10
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def accuracy_check(loader, model):
    model.eval()
    sm = F.softmax
    total, num_samples = 0, 0
    for images, labels in loader:
        labels, images = labels.to(device), images.to(device)
        outputs = model(images)
        sm_outputs = sm(outputs, dim=1)
        _, predicted = torch.max(sm_outputs.data, 1)
        total += (predicted == labels).sum().item()
        num_samples += labels.size(0)
    return round(100 * total / num_samples, 2)


def create_loss_fn(args):
    if args.method == 'mae':
        loss_fn = mae_loss
    elif args.method == 'mse':
        loss_fn = mse_loss
    elif args.method == 'ce':
        loss_fn = ce_loss
    elif args.method == 'gce':
        loss_fn = gce_loss
    elif args.method == 'phuber_ce':
        loss_fn = phuber_ce_loss
    elif args.method == 'log':
        loss_fn = log_loss
    elif args.method == 'exp':
        loss_fn = exp_loss
    else:
        loss_fn = None

    return loss_fn


def create_model(args, input_dim, input_channel, K):
    if args.model == 'mlp':
        model = mlp_model(input_dim=input_dim, hidden_dim=500, output_dim=K)
    elif args.model == 'linear':
        model = linear_model(input_dim=input_dim, output_dim=K)
    elif args.model == 'cnn':
        model = SmallCNN()
    elif args.model == 'resnet18':
        model = ResNet18(input_channel=input_channel, num_classes=K)
    elif args.model == 'densenet':
        model = DenseNet121(input_channel=input_channel, num_classes=K)
    elif args.model == "wrn":
        model = Wide_ResNet_Madry(depth=32, num_classes=K, widen_factor=10, dropRate=0.0)  # WRN-32-10

    return model


def get_pred(cl_model, data):
    cl_model.eval()
    with torch.no_grad():
        data = data.to(device)
        outputs = cl_model(data)
        _, predicted = torch.max(outputs, 1)

    return predicted, torch.softmax(outputs, dim=1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Learning with Complementary Labels')
    parser.add_argument('--cl_lr', type=float, default=5e-5, help='learning rate for complementary learning')
    parser.add_argument('--at_lr', type=float, default=1e-2, help='learning rate for adversarial training')
    parser.add_argument('--batch_size', type=int, default=256, help='batch_size of ordinary labels.')
    parser.add_argument('--cl_num', type=int, default=1, help='the number of complementary labels of each data.')
    parser.add_argument('--dataset', type=str, default="kuzushiji", choices=['mnist', 'kuzushiji', 'fashion', 'cifar10'],
                        help="dataset, choose from mnist, kuzushiji, fashion, cifar10")
    parser.add_argument('--method', type=str, default='exp', choices=['free', 'nn', 'ga', 'pc', 'forward', 'scl_exp',
                        'scl_nl', 'mae', 'mse', 'ce', 'gce', 'phuber_ce', 'log', 'exp', 'l_uw', 'l_w'])
    parser.add_argument('--model', type=str, default='cnn', choices=['linear', 'mlp', 'cnn', 'resnet18', 'densenet', 'wrn'], help='model name')
    parser.add_argument('--epochs', default=0, type=int, help='number of epochs for cl learning')
    parser.add_argument('--adv_epochs', default=100, type=int, help='number of epochs for adv')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
    parser.add_argument('--seed', type=int, nargs='+', default=[1, 2, 3], help='random seed')
    parser.add_argument('--out_dir', type=str, default='./ATCL_result', help='dir of output')
    # for adv training
    parser.add_argument('--epsilon', type=float, default=0.3, help='perturbation bound')
    parser.add_argument('--num_steps', type=int, default=40, help='maximum perturbation step K')
    parser.add_argument('--step_size', type=float, default=0.01, help='step size')
    parser.add_argument('--scheduler', type=str, default="linear", choices=['linear', 'cosine', 'none'], help='epsilon scheduler')
    parser.add_argument('--sch_epoch', type=int, default=50, help='scheduler epoch')
    parser.add_argument('--clat', action='store_true', help="do one-step pgd during complementary label training, default: False")
    args = parser.parse_args()

    # To be removed
    if args.dataset == "cifar10":
        args.cl_lr, args.at_lr, args.model, args.weight_decay, args.batch_size = 1e-2, 1e-1, 'resnet18', 5e-4, 128
        args.epsilon, args.num_steps, args.step_size = 0.031, 10, 0.007

    pp.pprint(vars(args))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    last_nature, last_pgd20, last_cw, best_nature, best_pgd20, best_cw = [], [], [], [], [], []
    for seed in args.seed:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        # Store path
        if not os.path.exists(args.out_dir):
            os.makedirs(args.out_dir)

        full_train_loader, train_loader, test_loader, ordinary_train_dataset, test_dataset, K, input_dim, input_channel = prepare_data(dataset=args.dataset, batch_size=args.batch_size)
        ordinary_train_loader, complementary_train_loader, ccp, x_to_mcls, x_to_tls, partialY = prepare_train_loaders(full_train_loader=full_train_loader, batch_size=args.batch_size, ordinary_train_dataset=ordinary_train_dataset, cl_num=args.cl_num)
        partialY = partialY.to(device)

        # complementary learning, ref to "Complementary-label learning for arbitrary losses and models"
        if args.epochs > 0:
            model = create_model(args, input_dim, input_channel, K)
            display_num_param(model)
            model = model.to(device)
            model = torch.nn.DataParallel(model)
            optimizer = torch.optim.SGD(model.parameters(), weight_decay=args.weight_decay, lr=args.cl_lr, momentum=args.momentum) if args.dataset == "cifar10" else torch.optim.Adam(model.parameters(), weight_decay=args.weight_decay, lr=args.cl_lr)
            train_accuracy = accuracy_check(loader=train_loader, model=model)
            test_accuracy = accuracy_check(loader=test_loader, model=model)
            print('Epoch: 0. Train_Set Acc: {}. Test_Set Acc: {}'.format(train_accuracy, test_accuracy))
            avg_test_acc = complementary_learning(args, model, optimizer, partialY, seed)

        # adversarial training with complementary labels
        if args.adv_epochs > 0:
            model = create_model(args, input_dim, input_channel, K)
            display_num_param(model)
            model = model.to(device)
            model = torch.nn.DataParallel(model)
            optimizer = torch.optim.SGD(model.parameters(), lr=args.at_lr, momentum=args.momentum)
            # last_res, best_res = two_stage_adversarial_train(args, model, optimizer, seed)
            last_res, best_res = one_stage_adversarial_train(args, model, optimizer, seed)

    print(last_nature);print(last_pgd20);print(last_cw);print(best_nature);print(best_pgd20);print(best_cw)
    print(">> Last Nature: {}({})".format(round(np.mean(last_nature), 2), round(np.std(last_nature, ddof=0), 2)))
    print(">> Last PGD20: {}({})".format(round(np.mean(last_pgd20), 2), round(np.std(last_pgd20, ddof=0), 2)))
    print(">> Last CW: {}({})".format(round(np.mean(last_cw), 2), round(np.std(last_cw, ddof=0), 2)))
    print(">> Best Nature: {}({})".format(round(np.mean(best_nature), 2), round(np.std(best_nature, ddof=0), 2)))
    print(">> Best PGD20: {}({})".format(round(np.mean(best_pgd20), 2), round(np.std(best_pgd20, ddof=0), 2)))
    print(">> Best CW: {}({})".format(round(np.mean(best_cw), 2), round(np.std(best_cw, ddof=0), 2)))
