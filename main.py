import argparse, time, os, random
from utils_data import *
from utils_algo import *
from models import *
from attack_generator import *


def adversarial_train(args, model, optimizer):
    lr, best_acc = args.lr, 0
    for epoch in range(args.epochs):
        lr = lr_schedule(lr, epoch + 1)
        optimizer.param_groups[0].update(lr=lr)
        for batch_idx, (images, cl_labels, true_labels, id) in enumerate(complementary_train_loader):
            random_cl_labels = []
            for i in id:
                mcls = x_to_mcls[i.item()]
                cl = list(mcls)[0] if len(mcls) == 1 else random.sample(mcls, 1)[0]
                random_cl_labels.append(cl)
            random_cl_labels = torch.LongTensor(random_cl_labels).to(device)
            images, cl_labels, true_labels = images.to(device), cl_labels.to(device), true_labels.to(device)

            # Get adversarial data
            x_adv, y_adv = adv_cl(model, images, random_cl_labels, true_labels, id, args.epsilon, args.step_size, args.num_steps, K, ccp, x_to_mcls,
                                  generate_cl_steps=args.generate_cl_steps, meta_method=args.method, category="Madry", rand_init=True)
            model.train()
            optimizer.zero_grad()
            logit = model(x_adv)
            loss, _ = chosen_loss_c(f=logit, K=K, labels=random_cl_labels, ccp=ccp, meta_method=args.method)
            loss.backward()
            optimizer.step()

            if batch_idx == 0:
                torch.set_printoptions(threshold=30000)
                print(y_adv)
                print()

        # test how many data are given wrong cls
        print(x_to_mcls)
        count = 0
        for k, v in x_to_mcls.items():
            if x_to_tls[k] in v: count += 1
        print("Epoch {}: {}/{}={}% data are given wrong complementary labels!".format(epoch+1, count, len(x_to_mcls), 100*count/len(x_to_mcls)))

        # Evalutions
        test_nat_acc = accuracy_check(loader=test_loader, model=model)
        _, test_pgd20_acc = eval_robust(model, test_loader, perturb_steps=20, epsilon=0.031, step_size=0.031 / 4, loss_fn="cent", category="Madry", random=True, num_classes=K)
        # _, cw_acc = eval_robust(model, test_loader, perturb_steps=30, epsilon=0.031, step_size=0.031 / 4, loss_fn="cw", category="Madry", random=True, num_classes=K)

        print('Epoch: [%d | %d] | Learning Rate: %f | Natural Test Acc %.4f | PGD20 Test Acc %.4f |\n' % (epoch+1, args.epochs, lr, test_nat_acc, test_pgd20_acc))

        # Save the last checkpoint
        torch.save({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'test_nat_acc': test_nat_acc,
            'test_pgd20_acc': test_pgd20_acc,
            'optimizer': optimizer.state_dict(),
        }, os.path.join(args.out_dir, "checkpoint.pth.tar"))


def complementary_learning(args, model, optimizer):
    lr = args.lr
    save_table = np.zeros(shape=(args.warmup_epochs, 3))
    for epoch in range(args.warmup_epochs):
        lr = lr_schedule(lr, epoch + 1)
        optimizer.param_groups[0].update(lr=lr)
        for i, (images, cl_labels, true_labels, id) in enumerate(complementary_train_loader):
            images, cl_labels, true_labels = images.to(device), cl_labels.to(device), true_labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
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
        print('Epoch: {}. Tr Acc: {}. Te Acc: {}.'.format(epoch + 1, train_accuracy, test_accuracy))
        save_table[epoch, :] = epoch + 1, train_accuracy, test_accuracy

    np.savetxt(args.method + '_results.txt', save_table, delimiter=',', fmt='%1.3f')


def lr_schedule(lr, epoch):
    if args.dataset == "cifar10" and epoch % 30 == 0:
        lr /= 2
    return lr


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Adversarial Training with Complementary Labels')
    parser.add_argument('--lr', type=float, default=1e-2, help='optimizer\'s learning rate', )
    parser.add_argument('--batch_size', type=int, default=256, help='batch_size of ordinary labels.')
    parser.add_argument('--dataset', type=str, default="cifar10", choices=['mnist', 'cifar10'],
                        help="dataset, choose from mnist, cifar10")
    parser.add_argument('--method', type=str, default='free', choices=['free', 'nn', 'ga', 'pc', 'forward'],
                        help='method type. ga: gradient ascent. nn: non-negative. free: Theorem 1. pc: Ishida2017. forward: Yu2018.')
    parser.add_argument('--model', type=str, default='resnet', choices=['linear', 'mlp', 'resnet'], help='model name',)
    parser.add_argument('--epochs', default=300, type=int, help='number of epochs')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--out_dir', type=str, default='./CLAT_result', help='dir of output')
    # for adv training
    parser.add_argument('--epsilon', type=float, default=0.031, help='perturbation bound')  # 0.3
    parser.add_argument('--num_steps', type=int, default=10, help='maximum perturbation step K')  # 40
    parser.add_argument('--step_size', type=float, default=0.007, help='step size')  # 0.01
    parser.add_argument('--generate_cl_steps', type=int, default=100, help='maximum step for generating multiple complementary labels, if <=0, skip it.')
    parser.add_argument('--warmup_epochs', default=0, type=int, help='number of cl warmup epochs')
    args = parser.parse_args()

    # To be removed
    args.epochs, args.warmup_epochs = 20, 20
    if args.dataset == "mnist":
        args.lr, args.model, args.weight_decay = 5e-5, 'mlp', 1e-4

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # Store path
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    full_train_loader, train_loader, test_loader, ordinary_train_dataset, test_dataset, K, input_dim = prepare_data(dataset=args.dataset, batch_size=args.batch_size)
    ordinary_train_loader, complementary_train_loader, ccp, x_to_mcls, x_to_tls = prepare_train_loaders(full_train_loader=full_train_loader, batch_size=args.batch_size, ordinary_train_dataset=ordinary_train_dataset)

    if args.model == 'mlp':
        model = mlp_model(input_dim=input_dim, hidden_dim=500, output_dim=K)
    elif args.model == 'linear':
        model = linear_model(input_dim=input_dim, output_dim=K)
    elif args.model == "resnet":
        model = ResNet34(num_classes=K)

    model = model.to(device)
    model = torch.nn.DataParallel(model)

    if args.dataset == "mnist":
        optimizer = torch.optim.Adam(model.parameters(), weight_decay=args.weight_decay, lr=args.lr)
    elif args.dataset == "cifar10":
        optimizer = torch.optim.SGD(model.parameters(), weight_decay=args.weight_decay, lr=args.lr, momentum=args.momentum)

    train_accuracy = accuracy_check(loader=train_loader, model=model)
    test_accuracy = accuracy_check(loader=test_loader, model=model)
    print('Epoch: 0. Train_Set Acc: {}. Test_Set Acc: {}'.format(train_accuracy, test_accuracy))

    # complementary learning, ref to "Complementary-label learning for arbitrary losses and models"
    if args.warmup_epochs > 0:
        complementary_learning(args, model, optimizer)

    adversarial_train(args, model, optimizer)
