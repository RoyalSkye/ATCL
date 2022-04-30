import argparse, time, os, random, math
import pprint as pp
import torchvision
from utils_data import *
from utils_algo import *
from utils_mcl_loss import *
from models import *
from attack_generator import *
from utils_func import *


def adversarial_train(args, model, epochs, mode="atcl", seed=1):
    print(">> Current mode: {} for {} epochs".format(mode, epochs))
    lr = args.cl_lr if mode == "cl" else args.at_lr
    loss_fn, best_nat_acc, best_pgd20_acc, best_cw_acc, best_epoch = create_loss_fn(args), 0, 0, 0, 0
    nature_train_acc_list, nature_test_acc_list, pgd20_acc_list, cw_acc_list = [], [], [], []
    first_layer_grad, last_layer_grad = [], []
    if mode == "cl":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.cl_lr, momentum=args.momentum, weight_decay=args.weight_decay) if args.dataset in ["cifar10", "svhn", "cifar100"] else torch.optim.Adam(model.parameters(), weight_decay=args.weight_decay, lr=args.cl_lr)
    elif mode == "at":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.at_lr, momentum=args.momentum, weight_decay=args.weight_decay) if args.dataset in ["cifar10", "svhn", "cifar100"] else torch.optim.SGD(model.parameters(), lr=args.at_lr, momentum=args.momentum)
        cl_model = create_model(args, input_dim, input_channel, K)
        checkpoint = torch.load(os.path.join(args.out_dir, "cl_best_checkpoint_seed{}.pth.tar".format(seed)))
        cl_model.load_state_dict(checkpoint['state_dict'])
        cl_model.eval()
        print(">> Load the CL model with train acc: {}, test acc: {}, epoch {}".format(checkpoint['train_acc'], checkpoint['test_acc'], checkpoint['epoch']))
    elif mode == "atcl":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.at_lr, momentum=args.momentum, weight_decay=args.weight_decay) if args.dataset in ["cifar10", "svhn", "cifar100"] else torch.optim.SGD(model.parameters(), lr=args.at_lr, momentum=args.momentum)

    for epoch in range(epochs):
        correct, total = 0, 0
        lr = cl_lr_schedule(lr, epoch + 1, optimizer) if mode == "cl" else adv_lr_schedule(lr, epoch + 1, optimizer)
        for batch_idx, (images, cl_labels, true_labels, id) in enumerate(complementary_train_loader):
            images, cl_labels, true_labels = images.to(device), cl_labels.to(device), true_labels.to(device)

            if mode == "cl":
                x_adv = images
            elif mode == "at":
                pseudo_labels, prob = get_pred(cl_model, images.detach())
                pseudo_labels = pseudo_labels.to(device)
                x_adv = pgd(model, images, pseudo_labels, args.epsilon, args.step_size, args.num_steps, loss_fn="cent", category="Madry", rand_init=True, num_classes=K)
            else:  # "atcl"
                _, prob = get_pred(model, images.detach())
                epsilon, step_size, num_steps = at_param_schedule(args, epoch + 1)
                alpha = atcl_scheduler(args, id, prob, epsilon, partialY, epoch + 1)
                _, pseudo_labels = torch.max(ema[id], 1)
                # pseudo_labels = torch.multinomial(ema[id], 1).squeeze(1)
                correct += (pseudo_labels == true_labels).sum().item()
                total += pseudo_labels.size(0)
                if epsilon != 0:
                    # x_adv = pgd(model, images, true_labels, epsilon, step_size, num_steps, loss_fn="cent", category="Madry", rand_init=True, num_classes=K)  # mex_ce_tl
                    # x_adv = pgd(model, images, None, epsilon, step_size, num_steps, loss_fn="kl", category="trades", rand_init=True, num_classes=K)  # max_trades
                    x_adv = cl_adv(args, model, images, cl_labels, epsilon, step_size, num_steps, id, ccp, partialY, pseudo_labels, alpha, loss_fn, category="Madry", rand_init=True)  # max_cl
                else:
                    x_adv = images

            if batch_idx == 0:
                torchvision.utils.save_image(x_adv, os.path.join(args.out_dir, "x_adv_seed_{}_epoch_{}.jpg".format(seed, epoch+1)))

            model.train()
            optimizer.zero_grad()
            logit = model(x_adv)
            if mode == "at":
                loss = nn.CrossEntropyLoss(reduction="mean")(logit, pseudo_labels)
            else:  # "cl" or "atcl"
                if args.method in ['exp', 'log']:
                    loss = loss_fn(logit, partialY[id].float())
                elif args.method in ['mae', 'mse', 'ce', 'gce', 'phuber_ce']:
                    loss = unbiased_estimator(loss_fn, logit, partialY[id].float())
                elif args.method in ['free', 'nn', 'ga', 'pc', 'forward', 'scl_exp', 'scl_nl', 'l_uw', 'l_w']:
                    assert args.cl_num == 1
                    loss, _ = chosen_loss_c(f=logit, K=K, labels=cl_labels, ccp=ccp, meta_method=args.method)
                elif args.method in ["log_ce"]:
                    loss = loss_fn(logit, partialY[id].float(), pseudo_labels, alpha)
            loss.backward()

            # the grad stat
            if args.model == "cnn":
                first_layer_grad.append(model.module.feature_extractor.conv1.weight.grad.norm(p=2).cpu().item())
                last_layer_grad.append(model.module.classifier.fc3.weight.grad.norm(p=2).cpu().item())
            elif args.model in ["densenet", "resnet18", "wrn"]:
                first_layer_grad.append(model.module.conv1.weight.grad.norm(p=2).cpu().item())
                last_layer_grad.append(model.module.linear.weight.grad.norm(p=2).cpu().item())
            else:
                assert True, "please modify grad stat code!"
            optimizer.step()

        # Evalutions
        model.eval()
        train_nat_acc = accuracy_check(loader=train_loader, model=model)
        test_nat_acc = accuracy_check(loader=test_loader, model=model)
        _, test_pgd20_acc = eval_robust(model, test_loader, perturb_steps=20, epsilon=args.epsilon, step_size=args.step_size, loss_fn="cent", category="Madry", random=True, num_classes=K)
        _, test_cw_acc = eval_robust(model, test_loader, perturb_steps=30, epsilon=args.epsilon, step_size=args.step_size, loss_fn="cw", category="Madry", random=True, num_classes=K)
        nature_train_acc_list.append(train_nat_acc);nature_test_acc_list.append(test_nat_acc);pgd20_acc_list.append(test_pgd20_acc);cw_acc_list.append(test_cw_acc)
        if mode == "atcl":
            print(round(correct/total*100, 2), alpha, epsilon, step_size, num_steps)
        print('Epoch: [%d | %d] | Learning Rate: %f | Natural Train Acc %.4f | Natural Test Acc %.4f | PGD20 Test Acc %.4f | CW Test Acc %.4f |\n' % (epoch + 1, epochs, lr, train_nat_acc, test_nat_acc, test_pgd20_acc, test_cw_acc))

        # Save the best & last checkpoint
        if mode == "cl":
            if test_nat_acc > best_nat_acc:
                best_epoch = epoch + 1
                best_nat_acc = test_nat_acc
                torch.save({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'train_acc': train_nat_acc,
                    'test_acc': test_nat_acc,
                    'optimizer': optimizer.state_dict(),
                }, os.path.join(args.out_dir, "cl_best_checkpoint_seed{}.pth.tar".format(seed)))
            torch.save({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'train_acc': train_nat_acc,
                'test_acc': test_nat_acc,
                'optimizer': optimizer.state_dict(),
            }, os.path.join(args.out_dir, "cl_checkpoint_seed{}.pth.tar".format(seed)))
        else:
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
            torch.save({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'test_nat_acc': test_nat_acc,
                'test_pgd20_acc': test_pgd20_acc,
                'test_cw_acc': test_cw_acc,
                'optimizer': optimizer.state_dict(),
            }, os.path.join(args.out_dir, "checkpoint_seed{}.pth.tar".format(seed)))

    if mode == "cl":
        print(nature_train_acc_list)
        print(nature_test_acc_list)
        print(">> Best test acc({}): {}".format(best_epoch, max(nature_test_acc_list)))
        print(">> AVG test acc of last 10 epochs: {}".format(np.mean(nature_test_acc_list[-10:])))
        epoch = [i for i in range(epochs)]
        show([epoch] * 2, [nature_train_acc_list, nature_test_acc_list], label=["train acc", "test acc"], title=args.dataset, xdes="Epoch", ydes="Accuracy", path=os.path.join(args.out_dir, "cl_acc_seed{}.png".format(seed)))

        return np.mean(nature_test_acc_list[-10:])
    else:
        print(nature_test_acc_list)
        print(pgd20_acc_list)
        print(cw_acc_list)
        print(">> Finished Adv Training: Natural Test Acc | Last_checkpoint %.4f | Best_checkpoint(%.1f) %.4f |\n" % (test_nat_acc, best_epoch, best_nat_acc))
        print(">> Finished Adv Training: PGD20 Test Acc | Last_checkpoint %.4f | Best_checkpoint %.4f |\n" % (test_pgd20_acc, best_pgd20_acc))
        print(">> Finished Adv Training: CW Test Acc | Last_checkpoint %.4f | Best_checkpoint %.4f |\n" % (test_cw_acc, best_cw_acc))
        epoch = [i for i in range(epochs)]
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
    if epoch <= args.warmup_epoch:
        return 0, 0, 0
    elif epoch <= (args.warmup_epoch+args.sch_epoch):
        if args.scheduler == "linear":
            eps = min(args.epsilon * ((epoch-args.warmup_epoch)/args.sch_epoch), args.epsilon)
        elif args.scheduler == "cosine":
            eps = 1/2 * (1-math.cos(math.pi * min(((epoch-args.warmup_epoch)/args.sch_epoch), 1))) * args.epsilon
        elif args.scheduler == "none":
            eps = args.epsilon
        # return eps, eps, 1
        return eps, args.step_size/args.epsilon*eps, args.num_steps
    else:
        return args.epsilon, args.step_size, args.num_steps


def atcl_scheduler(args, id, prob, epsilon, partialY, epoch):
    if args.sch_epoch == 0:
        alpha = 1 if epoch <= args.warmup_epoch else 0
    else:
        alpha = min(max(1 - (epoch-args.warmup_epoch)/args.sch_epoch, 0), 1)

    if epoch <= 5:
        ema[id] = ema[id]
    elif epsilon < args.epsilon/2:
        ema[id] = 0.9 * ema[id] + (1 - 0.9) * prob
    else:
        ema[id] = ema[id]

    ema[id] = ema[id] * partialY[id]  # reset to 0 for cls

    return alpha


def cl_lr_schedule(lr, epoch, optimizer):
    # no lr_decay for cll
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def adv_lr_schedule(lr, epoch, optimizer):
    if args.dataset in ["mnist", "fashion", "kuzushiji"]:
        # no lr_decay for small dataset
        pass
    elif args.dataset in ["cifar10", "svhn", "cifar100"]:
        if epoch == (30+args.warmup_epoch):
            lr /= 10
        if epoch == (60+args.warmup_epoch):
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
    elif args.method == 'log_ce':
        loss_fn = log_ce_loss
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
        model = densenet(input_channel=input_channel, num_classes=K)
    elif args.model == 'preact_resnet18':
        model = preactresnet18(input_channel=input_channel, num_classes=K)
    elif args.model == "wrn":
        model = Wide_ResNet_Madry(depth=32, num_classes=K, widen_factor=10, dropRate=0.0)  # WRN-32-10

    display_num_param(model)
    model = model.to(device)
    model = torch.nn.DataParallel(model)

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
    parser.add_argument('--cl_num', type=int, default=1, help='(1-9): the number of complementary labels of each data; (0): mul-cls data distribution of ICML2020')
    parser.add_argument('--dataset', type=str, default="mnist", choices=['mnist', 'kuzushiji', 'fashion', 'cifar10', 'svhn', 'cifar100'],
                        help="dataset, choose from mnist, kuzushiji, fashion, cifar10, svhn, cifar100")
    parser.add_argument('--framework', type=str, default='one_stage', choices=['one_stage', 'two_stage'])
    parser.add_argument('--method', type=str, default='exp', choices=['free', 'nn', 'ga', 'pc', 'forward', 'scl_exp',
                        'scl_nl', 'mae', 'mse', 'ce', 'gce', 'phuber_ce', 'log', 'exp', 'l_uw', 'l_w', 'log_ce'])
    parser.add_argument('--model', type=str, default='cnn', choices=['linear', 'mlp', 'cnn', 'resnet18', 'densenet', 'preact_resnet18', 'wrn'], help='model name')
    parser.add_argument('--cl_epochs', default=0, type=int, help='number of epochs for cl learning')
    parser.add_argument('--adv_epochs', default=100, type=int, help='number of epochs for adv')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
    parser.add_argument('--seed', type=int, nargs='+', default=[1, 2, 3], help='random seed')
    parser.add_argument('--out_dir', type=str, default='./ATCL_result', help='dir of output')
    # for adv training
    parser.add_argument('--epsilon', type=float, default=0.3, help='perturbation bound')
    parser.add_argument('--num_steps', type=int, default=40, help='maximum perturbation step K')
    parser.add_argument('--step_size', type=float, default=0.01, help='step size')
    parser.add_argument('--scheduler', type=str, default="none", choices=['linear', 'cosine', 'none'], help='epsilon scheduler')
    parser.add_argument('--sch_epoch', type=int, default=0, help='scheduler epoch')
    parser.add_argument('--warmup_epoch', type=int, default=0, help='warmup epoch for exponential moving average')
    args = parser.parse_args()

    # To be removed
    if args.dataset in ["cifar10", "svhn", "cifar100"]:
        args.weight_decay, args.batch_size = 5e-4, 128
        args.epsilon, args.num_steps, args.step_size = 8/255, 10, 2/255

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

        data_aug = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ]) if args.dataset in ["cifar10", "svhn", "cifar100"] else None
        train_loader, test_loader, ordinary_train_dataset, test_dataset, K, input_dim, input_channel = prepare_data(args)
        complementary_train_loader, ccp, partialY, ema = prepare_train_loaders(args, ordinary_train_dataset=ordinary_train_dataset, data_aug=data_aug)
        partialY, ema = partialY.to(device), ema.to(device)

        model = create_model(args, input_dim, input_channel, K)
        if args.framework == "two_stage":
            adversarial_train(args, model, args.cl_epochs, mode="cl", seed=seed)
            model = create_model(args, input_dim, input_channel, K)
            last_res, best_res = adversarial_train(args, model, args.adv_epochs, mode="at", seed=seed)
        else:
            last_res, best_res = adversarial_train(args, model, args.adv_epochs, mode="atcl", seed=seed)

        last_nature.append(last_res[0]);last_pgd20.append(last_res[1]);last_cw.append(last_res[2])
        best_nature.append(best_res[0]);best_pgd20.append(best_res[1]);best_cw.append(best_res[2])

    print(last_nature);print(last_pgd20);print(last_cw);print(best_nature);print(best_pgd20);print(best_cw)
    print(">> Last Nature: {}($\pm${})".format(round(np.mean(last_nature), 2), round(np.std(last_nature, ddof=0), 2)))
    print(">> Last PGD20: {}($\pm${})".format(round(np.mean(last_pgd20), 2), round(np.std(last_pgd20, ddof=0), 2)))
    print(">> Last CW: {}($\pm${})".format(round(np.mean(last_cw), 2), round(np.std(last_cw, ddof=0), 2)))
    print(">> Best Nature: {}($\pm${})".format(round(np.mean(best_nature), 2), round(np.std(best_nature, ddof=0), 2)))
    print(">> Best PGD20: {}($\pm${})".format(round(np.mean(best_pgd20), 2), round(np.std(best_pgd20, ddof=0), 2)))
    print(">> Best CW: {}($\pm${})".format(round(np.mean(best_cw), 2), round(np.std(best_cw, ddof=0), 2)))
