import argparse, time, os, random, math
from utils_data import *
from utils_algo import *
from utils_mcl_loss import *
from models import *
from attack_generator import *
from utils_func import *
import torchvision


def complementary_learning(args, model, optimizer, partialY, seed):
    loss_fn, loss_vector = create_loss_fn(args), None
    lr, best_acc, best_epoch, train_acc_list, test_acc_list = args.lr, 0, 0, [], []
    for epoch in range(args.epochs):
        lr = lr_schedule(lr, epoch + 1, optimizer)
        for i, (images, cl_labels, true_labels, id) in enumerate(complementary_train_loader):
            images, cl_labels, true_labels = images.to(device), cl_labels.to(device), true_labels.to(device)
            optimizer.zero_grad()
            if args.at:
                x_adv, _ = adv_cl(args, model, images, cl_labels, true_labels, id, ccp, partialY, loss_fn, category="Madry", rand_init=True)
                outputs = model(x_adv)
            else:
                outputs = model(images)

            if args.method in ['exp', 'log']:
                loss = loss_fn(outputs, partialY[id].float())
            elif args.method in ['mae', 'mse', 'ce', 'gce', 'phuber_ce']:
                loss = unbiased_estimator(loss_fn, outputs, partialY[id].float())
            elif args.cl_num == 1 and args.method in ['free', 'nn', 'ga', 'pc', 'forward', 'scl_exp', 'scl_nl']:
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

    # stat
    wrong_count, correct_count = stat(x_to_mcls, x_to_tls)
    print("Epoch {}: {}% data are given wrong complementary labels, and each data are given {} correct complementary labels on average!".format(epoch + 1, wrong_count * 100, correct_count))

    # plot
    print(train_acc_list)
    print(test_acc_list)
    print(">> Best test acc({}): {}".format(best_epoch, max(test_acc_list)))
    epoch = [i for i in range(args.epochs)]
    show([epoch] * 2, [train_acc_list, test_acc_list], label=["train acc", "test acc"], title=args.dataset, xdes="Epoch", ydes="Accuracy", path=os.path.join(args.out_dir, "cl_acc_seed{}.png".format(seed)))


def lr_schedule(lr, epoch, optimizer):
    if args.dataset == "cifar10" and epoch % 30 == 0:
        lr /= 2
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def accuracy_check(loader, model):
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
        model = cnn(input_channel=input_channel, num_classes=K)
    elif args.model == 'resnet18':
        model = ResNet18(input_channel=input_channel, num_classes=K)
    elif args.model == 'resnet34':
        model = ResNet34(input_channel=input_channel, num_classes=K)
    elif args.model == 'densenet':
        model = densenet(input_channel=input_channel, num_classes=K)
    elif args.model == "wrn":
        model = Wide_ResNet_Madry(depth=32, num_classes=K, widen_factor=10, dropRate=0.0)  # WRN-32-10

    return model


def get_pred(cl_model, data):
    cl_model.eval()
    with torch.no_grad():
        data = data.to(device)
        outputs = cl_model(data)
        _, predicted = torch.max(outputs, 1)

    return predicted


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Learning with Complementary Labels')
    parser.add_argument('--lr', type=float, default=5e-5, help='optimizer\'s learning rate', )
    parser.add_argument('--batch_size', type=int, default=256, help='batch_size of ordinary labels.')
    parser.add_argument('--cl_num', type=int, default=1, help='the number of complementary labels of each data.')
    parser.add_argument('--dataset', type=str, default="mnist", choices=['mnist', 'kuzushiji', 'fashion', 'cifar10'],
                        help="dataset, choose from mnist, kuzushiji, fashion, cifar10")
    parser.add_argument('--method', type=str, default='exp', choices=['free', 'nn', 'ga', 'pc', 'forward', 'scl_exp',
                        'scl_nl', 'mae', 'mse', 'ce', 'gce', 'phuber_ce', 'log', 'exp'])
    parser.add_argument('--model', type=str, default='mlp', choices=['linear', 'mlp', 'cnn', 'resnet18', 'resnet34', 'densenet', 'wrn'], help='model name')
    parser.add_argument('--epochs', default=300, type=int, help='number of epochs')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
    parser.add_argument('--seed', type=int, nargs='+', default=[1, 2, 3], help='random seed')
    parser.add_argument('--out_dir', type=str, default='./ATCL_result', help='dir of output')
    # for adv training
    parser.add_argument('--epsilon', type=float, default=0.007, help='perturbation bound')
    parser.add_argument('--num_steps', type=int, default=1, help='maximum perturbation step K')
    parser.add_argument('--step_size', type=float, default=0.007, help='step size')
    parser.add_argument('--at', action='store_true', help="do one-step pgd during training, default: False")
    args = parser.parse_args()

    # To be removed
    if args.dataset == "cifar10":
        args.lr, args.model, args.weight_decay = 1e-2, 'resnet34', 5e-4

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    for seed in args.seed:
        print(">> dataset: {}, cl_num: {}, model: {}, lr: {}, weight_decay: {}, seed: {}".format(args.dataset, args.cl_num, args.model, args.lr, args.weight_decay, seed))
        if args.at:
            print(">> epsilon: {}, step_size: {}, num_steps: {}".format(args.epsilon, args.step_size, args.num_steps))
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

        model = create_model(args, input_dim, input_channel, K)
        print(args.model)
        display_num_param(model)
        model = model.to(device)
        model = torch.nn.DataParallel(model)

        if args.dataset == "cifar10":
            optimizer = torch.optim.SGD(model.parameters(), weight_decay=args.weight_decay, lr=args.lr, momentum=args.momentum)
        else:
            optimizer = torch.optim.Adam(model.parameters(), weight_decay=args.weight_decay, lr=args.lr)

        train_accuracy = accuracy_check(loader=train_loader, model=model)
        test_accuracy = accuracy_check(loader=test_loader, model=model)
        print('Epoch: 0. Train_Set Acc: {}. Test_Set Acc: {}'.format(train_accuracy, test_accuracy))

        # complementary learning, ref to "Complementary-label learning for arbitrary losses and models"
        print(">> Learning with Complementary Labels")
        complementary_learning(args, model, optimizer, partialY, seed)
