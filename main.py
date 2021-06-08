from utils.utils import get_data, load_data, load_and_reformat_image, get_target_pairs
from models.SiameseNetwork import SiameseNetwork
from glob import glob
from collections import Counter
import numpy as np
import torch
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def data_generator(selector):
    if selector == "train":
        pos_target_pairs = train_pos_target_pairs
        neg_target_pairs = train_neg_target_pairs
        n_batch = n_batch_train
        images = train_images

    if selector == "test":
        pos_target_pairs = test_pos_target_pairs
        neg_target_pairs = test_neg_target_pairs
        n_batch = n_batch_test
        images = test_images

    while True:
        # shuffle the dataset each epoch
        pos_pairs_idx = shuffle(pos_target_pairs)
        neg_pairs_idx = shuffle(neg_target_pairs)

        for n in range(n_batch):
            # we devide the batch_size in 2 because we later concatenate it
            # the index_pairs are imbalanced, we make it balanced - every batch
            # will be coposed of 50% negative pairs and 50% positive pairs

            neg_img_pairs_idx = neg_pairs_idx[n * batch_size // 2:(n + 1) * batch_size // 2]
            neg_img_pairs = images[neg_img_pairs_idx]

            pos_pairs_batch_idx = pos_pairs_idx[n * batch_size // 2:(n + 1) * batch_size // 2]
            pos_img_pairs = images[pos_pairs_batch_idx]

            # we label the positive pairs as 1, and the negative pairs as 0
            image_pairs = np.concatenate([pos_img_pairs, neg_img_pairs])
            labels_batch = np.concatenate([[1] * (batch_size // 2), [0] * (batch_size // 2)])

            image_pairs, labels_batch = shuffle(image_pairs, labels_batch)

            image_batch_1 = image_pairs[:, 0, :, :]
            image_batch_2 = image_pairs[:, 1, :, :]

            yield image_batch_1, image_batch_2, labels_batch


def calc_loss(distance_batch, label_batch):
    # if the label is 1 (positive pairs), the loss function will want to minimize the distance between the image embedding
    # if the label is 0 (negative pairs), the loss function will want to maximizer the distance between the image embedding
    # the max function in the lost function second term is to deal with the case the distance is bigger then 1

    zeros = torch.zeros_like(distance_batch)
    loss = label_batch * distance_batch ** 2 + (1 - label_batch) * (torch.max(1 - distance_batch, zeros)) ** 2
    loss = torch.mean(loss)
    return loss


# calc the accuracy for selected treshold. the treshold we choose will seperate between matched and non-matched face pairs
def calc_accuracy(selector, treshold):
    if selector == 'Train':
        neg_images_pairs = train_neg_target_pairs
        pos_images_pairs = train_pos_target_pairs
        images = train_images

    if selector == 'Test':
        neg_images_pairs = test_neg_target_pairs
        pos_images_pairs = test_pos_target_pairs
        images = test_images

    # the distance array that will be used for the histogrm plot
    neg_dists = []
    pos_dists = []

    # the values that will be used for the sensitivity and specificity calculation
    true_negatives = 0
    false_positives = 0
    true_positives = 0
    false_negatives = 0

    # we will devide the data into batches for faster calculation
    n_batches = len(neg_images_pairs) // batch_size

    # first calc the distance for the negative pairs
    for n in range(n_batches):
        negative_batch = np.array(images)[neg_images_pairs[n * batch_size:(n + 1) * batch_size]]

        img1_batch = torch.tensor(negative_batch[:, 0]).to(device)
        img2_batch = torch.tensor(negative_batch[:, 1]).to(device)

        # the distance predictions for each batch
        dist = model(img1_batch, img2_batch).cpu().detach().numpy()

        neg_dists = np.concatenate((neg_dists, dist))

        # sum up the correct and false predictions for the negative pairs
        true_negatives += np.sum(neg_dists >= treshold)
        false_positives += np.sum(neg_dists < treshold)

    # calc the distance for the positive pairs
    n_batches = len(pos_images_pairs) // batch_size

    for n in range(n_batches):
        positive_batch = np.array(images)[pos_images_pairs[n * batch_size:(n + 1) * batch_size]]
        img1_batch = torch.tensor(positive_batch[:, 0]).to(device)
        img2_batch = torch.tensor(positive_batch[:, 1]).to(device)

        dist = model(img1_batch, img2_batch).cpu().detach().numpy()
        pos_dists = np.concatenate((pos_dists, dist))

        true_positives += np.sum(pos_dists < treshold)
        false_negatives += np.sum(pos_dists >= treshold)

    # the radio of correctly predicted matched pairs to the total number of matched pairs
    sensitivity = true_positives / (true_positives + false_negatives)

    # the radio of correctly predicted non-matched pairs to the total number of non-matched pairs
    specificity = true_negatives / (true_negatives + false_positives)

    return pos_dists, neg_dists, sensitivity, specificity


def train():
    model.train()
    epoch = 0
    train_losses = []
    test_losses = []

    epochs = 20
    i = 0
    train_epoch_losses = []
    test_epoch_losses = []
    for img1_batch, img2_batch, labels_batch in data_generator("train"):
        model.train()
        img1_batch = torch.tensor(img1_batch).to(device)
        img2_batch = torch.tensor(img2_batch).to(device)
        labels_batch = torch.tensor(labels_batch, dtype=torch.float32).to(device)

        optimizer.zero_grad()
        # calculate the distance between the images
        distance_batch = model(img1_batch, img2_batch)

        # calc the loss of the batch
        train_loss = calc_loss(distance_batch, labels_batch)
        train_epoch_losses.append(train_loss.item())
        # compute gradients and performing gradient descend

        train_loss.backward()
        optimizer.step()

        model.eval()
        # calculate the lost for one batch of the test data
        for img1_batch, img2_batch, labels_batch in data_generator("test"):
            img1_batch = torch.tensor(img1_batch).to(device)
            img2_batch = torch.tensor(img2_batch).to(device)
            labels_batch = torch.tensor(labels_batch, dtype=torch.float32).to(device)

            distance_batch = model(img1_batch, img2_batch)

            test_loss = calc_loss(distance_batch, labels_batch)
            test_epoch_losses.append(test_loss.item())
            break

            # for later loss plot

        # because we getting the data from generator we need to check when the epoch is ending
        i += 1
        if i % n_batch_train == 0:
            epoch += 1
            i = 0
            train_epoch_loss = np.mean(train_epoch_losses)
            train_losses.append(train_epoch_loss)

            test_epoch_loss = np.mean(test_epoch_losses)
            test_losses.append(test_epoch_loss)

            train_epoch_losses = []
            test_epoch_losses = []
            print(f'Epoch {epoch}/{epochs}, Train Loss: {train_epoch_loss:.4f}, \
          Test Loss: {test_epoch_loss:.4f}')

        if epoch == epochs:
            break
    return train_losses, test_losses


if __name__ == "__main__":
    get_data()
    # image dims
    H = 60
    W = 80
    filenames = glob('data/yalefaces/*')
    n = len(filenames)  # number of samples

    images, labels = load_data(filenames)

    samples_per_label = Counter(labels)

    num_of_labels = len(set(labels))  # 15 different persons
    samples_per_label = np.array([samples_per_label[i] for i in range(num_of_labels)])

    # split the data into train and test sets. each of the of the sets will contain similar proportion of labels
    train_samples_per_label = np.uint8(samples_per_label * 0.8)

    num_of_train_samples = sum(train_samples_per_label)
    num_of_test_samples = n - num_of_train_samples

    # initialize arrays to hold train and test images and labels
    train_images = np.empty((num_of_train_samples, 1, H, W), dtype=np.float32)
    test_images = np.empty((num_of_test_samples, 1, H, W), dtype=np.float32)
    train_labels = np.empty(num_of_train_samples, dtype=np.float32)
    test_labels = np.empty(num_of_test_samples, dtype=np.float32)

    count_so_far = {}
    train_idx = 0
    test_idx = 0
    # images, labels = shuffle(images, labels)
    for img, label in zip(images, labels):
        # increment the count
        count_so_far[label] = count_so_far.get(label, 0) + 1

        if count_so_far[label] > 3:
            # we have already added 3 test images for this subject
            # so add the rest to train
            train_images[train_idx] = img
            train_labels[train_idx] = label
            train_idx += 1

        else:
            # add the first 3 images to test
            test_images[test_idx] = img
            test_labels[test_idx] = label
            test_idx += 1

        train_pos_target_pairs, train_neg_target_pairs = get_target_pairs(train_labels)
        test_pos_target_pairs, test_neg_target_pairs = get_target_pairs(test_labels)

        batch_size = 32
        N_train_pos = len(train_pos_target_pairs) * 2
        n_batch_train = N_train_pos // batch_size

        N_test_pos = len(test_pos_target_pairs) * 2
        n_batch_test = N_test_pos // batch_size

    # embedding dim of 50
    model = SiameseNetwork(50).to(device)
    optimizer = torch.optim.Adam(model.parameters())

    #start the train loop
    train_losses, test_losses = train()

    plt.plot(train_losses, label="Train loss")
    plt.plot(test_losses, label="Test loss")
    plt.title("Loss per epoch")
    plt.legend()
    plt.show()

    # insert the model into evaluate mode
    model.eval()

    # treshold value of 0.65 seems to be reasonable according to the histogram
    train_pos_dists, train_neg_dists, train_sensitivity, train_specificity = calc_accuracy('Train', treshold=0.71)
    test_pos_dists, test_neg_dists, test_sensitivity, test_specificity = calc_accuracy('Test', treshold=0.71)

    print("Train sensitivity=", train_sensitivity)
    print("Train specificity=", train_specificity)
    print("")
    print("Test sensitivity=", test_sensitivity)
    print("Test specificity=", test_specificity)
    # we want to plot the distance distribution, in order to define where will be the treshold,
    # that seperates between positive and negative predictions
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))

    plt.suptitle("Distance distribution")

    ax[0].hist(train_pos_dists, bins=20, density=True, label="Positive distance")
    ax[0].hist(train_neg_dists, bins=20, density=True, label="Negative distance")
    ax[0].set_title("Train distance distribution")
    ax[0].legend()

    ax[1].hist(test_pos_dists, bins=20, density=True, label="Positive distance")
    ax[1].hist(test_neg_dists, bins=20, density=True, label="Negative distance")
    ax[1].set_title("Test distance distribution")
    ax[1].legend()

    plt.show()

    # lets try different reshold values to see where we gets the desired result
    # the best values is relative to the result we wish to get.
    # if we care the most about correctly predicted similar faces, then we will choose treshold that gives high sensetivity,
    # however it will lower the specificity
    # if we care the most about correctly predicted non-similar face, then we will choose treshold that gives high specificity,
    # however it will lower the senitivity

    treshold_values = np.linspace(0.5, 0.9, 20)
    accuracy = 0
    for treshold in treshold_values:
        _, _, test_sensitivity, test_specificity = calc_accuracy('Test', treshold)
        print(f'Treshold= {treshold:.4f}')
        print(f'Test sensitivity={test_sensitivity:.4f}')
        print(f'Test specificity={test_specificity:.4f}')
        print("")

    # for the project purpose, let's choose treshold of 0.71
    # lets randomly choose images pair for the test set and see the prediction
    input_value = 'y'
    treshold = 0.71
    while input_value == 'y':
        if np.random.random() > 0.5:
            rand_idx = np.random.randint(len(test_pos_target_pairs))
            chosen_idx = test_pos_target_pairs[rand_idx]

        else:
            rand_idx = np.random.randint(len(test_neg_target_pairs))
            chosen_idx = test_neg_target_pairs[rand_idx]

        img1 = test_images[chosen_idx][0]
        img2 = test_images[chosen_idx][1]
        _, ax = plt.subplots(1, 2)
        ax[0].imshow(img1[0], cmap='gray')
        ax[1].imshow(img2[0], cmap='gray')
        plt.show()

        img1 = torch.tensor(img1)
        img2 = torch.tensor(img2)

        img1 = torch.unsqueeze(img1, 0).to(device)
        img2 = torch.unsqueeze(img2, 0).to(device)

        dist = model(img1, img2).cpu().detach().numpy()
        if dist < treshold:
            prediction = 'Face match'
        else:
            prediction = 'Face mismatch'
        print(f'{prediction} (distance={dist[0]:.4f})')

        input_value = input("Enter 'y' for another image pairs, or 'n' to exit the loop:")
