import argparse, time, os, random, math
from utils_data import *
from utils_algo import *
from utils_mcl_loss import *
from models import *
from attack_generator import *
from utils_func import *
import torchvision


def adversarial_train(args, model, optimizer, partialY):
    lr, cl_model, best_pgd20_acc, nature_test_acc_list, pgd20_acc_list = args.lr, None, 0, [], []
    if args.baseline == "two_stage":
        cl_model = torch.nn.DataParallel(create_model(args, input_dim, input_channel, K).to(device))
        checkpoint = torch.load(os.path.join(args.out_dir, "cl_best_checkpoint.pth.tar"))
        cl_model.load_state_dict(checkpoint['state_dict'])
        cl_model.eval()
        print(">> Load the best CL model with test acc: {}, epoch {}".format(checkpoint['test_acc'], checkpoint['epoch']))
    for epoch in range(args.epochs):
        lr = lr_schedule(lr, epoch + 1)
        optimizer.param_groups[0].update(lr=lr)
        for batch_idx, (images, cl_labels, true_labels, id) in enumerate(complementary_train_loader):
            # for two-stage baseline
            if args.baseline == "two_stage":
                pseudo_labels = get_pred(cl_model, images.detach())
                images, pseudo_labels = images.to(device), pseudo_labels.to(device)
                x_adv = pgd(model, images, pseudo_labels, args.epsilon, args.step_size, args.num_steps, loss_fn="cent", category="Madry", rand_init=True, num_classes=K)
                model.train()
                optimizer.zero_grad()
                logit = model(x_adv)
                loss = nn.CrossEntropyLoss(reduction="mean")(logit, pseudo_labels)
                loss.backward()
                optimizer.step()
                continue

            # random_cl_labels = []
            # for i in id:
            #     mcls = x_to_mcls[i.item()]
            #     cl = list(mcls)[0] if len(mcls) == 1 else random.sample(mcls, 1)[0]
            #     random_cl_labels.append(cl)
            # random_cl_labels = torch.LongTensor(random_cl_labels).to(device)
            images, cl_labels, true_labels = images.to(device), cl_labels.to(device), true_labels.to(device)

            # Get adversarial data
            # num_steps = math.ceil((epoch+1) / args.epochs * args.num_steps) if args.progressive else args.num_steps
            epsilon = (epoch+1) / args.epochs * args.epsilon if args.progressive else args.epsilon
            x_adv, y_adv, partialY = adv_cl(model, images, cl_labels, pseudo_labels, true_labels, id, epsilon, args.step_size, args.num_steps, K, ccp, x_to_mcls, partialY,
                                  generate_cl_steps=args.generate_cl_steps, meta_method=args.method, category="Madry", rand_init=True)
            model.train()
            optimizer.zero_grad()
            logit = model(x_adv)

            # cl_w = weight_update(cl_w, logit.detach(), id)
            # loss, loss_vector = chosen_loss_c(f=logit, K=K, labels=cl_labels, ccp=ccp, meta_method=args.method)
            # loss, loss_vector = weighted_mcl_loss(logit, cl_w[id], ccp, args.method)
            loss = exp_loss(logit, partialY[id].float())
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

            if batch_idx == 0:
                torchvision.utils.save_image(x_adv, os.path.join(args.out_dir, "x_adv_epoch_{}.jpg".format(epoch+1)))
                torch.set_printoptions(threshold=600000)
                # print(y_adv)
                print()

        # stat
        wrong_count, correct_count = stat(x_to_mcls, x_to_tls)
        print("Epoch {}: {}% data are given wrong complementary labels, and each data are given {} correct complementary labels on average!".format(epoch + 1, wrong_count * 100, correct_count))

        # Evalutions
        test_nat_acc = accuracy_check(loader=test_loader, model=model)
        _, test_pgd20_acc = eval_robust(model, test_loader, perturb_steps=20, epsilon=args.epsilon, step_size=args.step_size, loss_fn="cent", category="Madry", random=True, num_classes=K)
        nature_test_acc_list.append(test_nat_acc)
        pgd20_acc_list.append(test_pgd20_acc)

        print('Epoch: [%d | %d] | Learning Rate: %f | Natural Test Acc %.4f | PGD20 Test Acc %.4f |\n' % (epoch+1, args.epochs, lr, test_nat_acc, test_pgd20_acc))

        # Save the best checkpoint
        if test_pgd20_acc > best_pgd20_acc:
            best_pgd20_acc = test_pgd20_acc
            torch.save({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'test_nat_acc': test_nat_acc,
                'test_pgd20_acc': test_pgd20_acc,
                'optimizer': optimizer.state_dict(),
            }, os.path.join(args.out_dir, "best_checkpoint.pth.tar"))

        # Save the last checkpoint
        torch.save({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'test_nat_acc': test_nat_acc,
            'test_pgd20_acc': test_pgd20_acc,
            'optimizer': optimizer.state_dict(),
        }, os.path.join(args.out_dir, "checkpoint.pth.tar"))

    print(nature_test_acc_list)
    print(pgd20_acc_list)
    print(">> Finished Adv Training: PGD20 Test Acc | Last_checkpoint %.4f | Best_checkpoint %.4f |\n" % (test_pgd20_acc, best_pgd20_acc))

    # plot
    epoch = [i for i in range(args.epochs)]
    show([epoch, epoch], [nature_test_acc_list, pgd20_acc_list], label=["nature acc", "pgd20 acc"], title=args.dataset,
         xdes="Epoch", ydes="Test Accuracy", path=os.path.join(args.out_dir, "adv_test_acc.png"))

    # eval using CW Attack
    model.eval()
    model_path, best_model_path = os.path.join(args.out_dir, "checkpoint.pth.tar"), os.path.join(args.out_dir, "best_checkpoint.pth.tar")
    checkpoint, best_checkpoint = torch.load(model_path), torch.load(best_model_path)
    model.load_state_dict(checkpoint['state_dict'])
    _, last_cw_acc = eval_robust(model, test_loader, perturb_steps=30, epsilon=args.epsilon, step_size=args.step_size, loss_fn="cw", category="Madry", random=True, num_classes=K)
    model.load_state_dict(best_checkpoint['state_dict'])
    _, best_cw_acc = eval_robust(model, test_loader, perturb_steps=30, epsilon=args.epsilon, step_size=args.step_size, loss_fn="cw", category="Madry", random=True, num_classes=K)
    print(">> EVAL: CW Test Acc | Last_checkpoint %.4f | Best_checkpoint(%.1f) %.4f |\n" % (last_cw_acc, best_checkpoint['epoch'], best_cw_acc))


def complementary_learning(args, model, optimizer, partialY):
    lr, best_acc, train_acc_list, test_acc_list = args.lr, 0, [], []
    for epoch in range(args.warmup_epochs):
        lr = lr_schedule(lr, epoch + 1)
        optimizer.param_groups[0].update(lr=lr)
        for i, (images, cl_labels, true_labels, id) in enumerate(complementary_train_loader):
            images, cl_labels, true_labels = images.to(device), cl_labels.to(device), true_labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            # loss, loss_vector = chosen_loss_c(f=outputs, K=K, labels=cl_labels, ccp=ccp, meta_method=args.method)
            loss = exp_loss(outputs, partialY[id].float())
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
        print('Epoch: {}. Tr Acc: {}. Te Acc: {}.'.format(epoch + 1, train_accuracy, test_accuracy))

        # Save the best checkpoint
        if test_accuracy > best_acc:
            best_acc = test_accuracy
            torch.save({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'test_acc': test_accuracy,
                'optimizer': optimizer.state_dict(),
            }, os.path.join(args.out_dir, "cl_best_checkpoint.pth.tar"))

    # stat
    wrong_count, correct_count = stat(x_to_mcls, x_to_tls)
    print("Epoch {}: {}% data are given wrong complementary labels, and each data are given {} correct complementary labels on average!".format(epoch + 1, wrong_count * 100, correct_count))

    # plot
    print(train_acc_list)
    print(test_acc_list)
    print(">> Best test acc: {}".format(max(test_acc_list)))
    epoch = [i for i in range(args.warmup_epochs)]
    show([epoch, epoch], [train_acc_list, test_acc_list], label=["train set acc", "test set acc"], title=args.dataset,
         xdes="Epoch", ydes="Accuracy", path=os.path.join(args.out_dir, "cl_acc.png"))


def lr_schedule(lr, epoch):
    if args.dataset == "cifar10" and epoch % 30 == 0:
        lr /= 2
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
    return 100 * total / num_samples


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


def weight_update(weight, logit, id):
    weight = weight.to(device)
    # num_class = logit.size(1)
    # for i, pred in enumerate(logit):
    #     idx = id[i].item()
    #     mcls = x_to_mcls[idx]
    #     weight[idx] = torch.zeros(num_class).scatter_(0, torch.LongTensor(list(mcls)), 1) / len(mcls)
    new_weight = 1 - torch.softmax(logit, dim=1)
    new_weight = new_weight / torch.sum(new_weight, dim=1).view(-1, 1)
    weight[id] = args.alpha * weight[id] + (1 - args.alpha) * new_weight

    return weight


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Adversarial Training with Complementary Labels')
    parser.add_argument('--lr', type=float, default=1e-2, help='optimizer\'s learning rate', )
    parser.add_argument('--batch_size', type=int, default=256, help='batch_size of ordinary labels.')
    parser.add_argument('--dataset', type=str, default="cifar10", choices=['mnist', 'kuzushiji', 'fashion', 'cifar10'],
                        help="dataset, choose from mnist, kuzushiji, fashion, cifar10")
    parser.add_argument('--method', type=str, default='free', choices=['free', 'nn', 'ga', 'pc', 'forward', 'scl_exp', 'scl_nl'],
                        help='method type. ga: gradient ascent. nn: non-negative. free: Theorem 1. pc: Ishida2017. forward: Yu2018.')
    parser.add_argument('--model', type=str, default='resnet34', choices=['linear', 'mlp', 'cnn', 'resnet18', 'resnet34', 'densenet', 'wrn'], help='model name')
    parser.add_argument('--baseline', type=str, default='two_stage', choices=['one_stage', 'two_stage'], help='baseline methods')
    parser.add_argument('--epochs', default=300, type=int, help='number of epochs')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
    parser.add_argument('--alpha', type=float, default=0.9, help='exponential moving average')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--out_dir', type=str, default='./ATCL_result', help='dir of output')
    # for adv training
    parser.add_argument('--epsilon', type=float, default=0.031, help='perturbation bound')
    parser.add_argument('--num_steps', type=int, default=10, help='maximum perturbation step K')
    parser.add_argument('--step_size', type=float, default=0.007, help='step size')
    parser.add_argument('--generate_cl_steps', type=int, default=0, help='maximum step for generating multiple complementary labels, if <=0, skip it.')
    parser.add_argument('--warmup_epochs', default=0, type=int, help='number of cl warmup epochs')
    parser.add_argument('--progressive', action='store_true', help="progressively increase the num_steps/epsilon of PGD, default: False")
    args = parser.parse_args()

    # To be removed
    args.epochs, args.warmup_epochs, args.generate_cl_steps = 300, 0, 0
    if args.dataset != "cifar10":
        args.lr, args.model, args.weight_decay = 5e-5, 'mlp', 1e-4
        args.epsilon, args.num_steps, args.step_size = 0.3, 40, 0.01

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

    full_train_loader, train_loader, test_loader, ordinary_train_dataset, test_dataset, K, input_dim, input_channel = prepare_data(dataset=args.dataset, batch_size=args.batch_size)
    ordinary_train_loader, complementary_train_loader, ccp, x_to_mcls, x_to_tls, partialY = prepare_train_loaders(full_train_loader=full_train_loader, batch_size=args.batch_size, ordinary_train_dataset=ordinary_train_dataset)
    partialY = partialY.to(device)

    model = create_model(args, input_dim, input_channel, K)
    print(args.model)
    display_num_param(model)
    model = model.to(device)
    model = torch.nn.DataParallel(model)

    if args.dataset == "cifar10":
        optimizer1 = torch.optim.SGD(model.parameters(), weight_decay=args.weight_decay, lr=args.lr,momentum=args.momentum)
    else:
        optimizer1 = torch.optim.Adam(model.parameters(), weight_decay=args.weight_decay, lr=args.lr)

    train_accuracy = accuracy_check(loader=train_loader, model=model)
    test_accuracy = accuracy_check(loader=test_loader, model=model)
    print('Epoch: 0. Train_Set Acc: {}. Test_Set Acc: {}'.format(train_accuracy, test_accuracy))

    # complementary learning, ref to "Complementary-label learning for arbitrary losses and models"
    if args.warmup_epochs > 0:
        print(">> Learning with Complementary Labels")
        complementary_learning(args, model, optimizer1, partialY)

    # for two_stage baseline
    adv_model = torch.nn.DataParallel(create_model(args, input_dim, input_channel, K).to(device)) if args.baseline == "two_stage" else model
    if args.dataset == "cifar10":
        optimizer2 = torch.optim.SGD(adv_model.parameters(), weight_decay=args.weight_decay, lr=args.lr, momentum=args.momentum) if args.baseline == "two_stage" else optimizer1
    else:
        optimizer2 = torch.optim.Adam(adv_model.parameters(), weight_decay=args.weight_decay, lr=args.lr) if args.baseline == "two_stage" else optimizer1

    if args.epochs > 0:
        print(">> Adversarial learning with Complementary Labels")
        adversarial_train(args, adv_model, optimizer2, partialY)
