"""
ModelTester class

@author: Hugues THOMAS, Oslandia
@date: july 2024
"""

import os
from pathlib import Path
import time

import numpy as np
import torch

from kpconv_torch.utils import colors
from kpconv_torch.utils.metrics import fast_confusion, IoU_from_confusions
from kpconv_torch.io.ply import write_ply
from kpconv_torch.io import ply


def get_test_save_path(infered_file: Path, chosen_log: Path) -> Path:
    if chosen_log is None:
        test_path = None
    elif infered_file is not None:
        test_path = Path(infered_file).parent / "test" / Path(chosen_log).name
    else:
        test_path = Path(chosen_log) / "test"
    if test_path is not None and not os.path.exists(test_path):
        os.makedirs(test_path)
    return test_path


class ModelTester:
    def __init__(self, net, config, chkp_path=None, test_path=None, on_gpu=True):
        """
        Initialize training parameters and reload previous model for restore/finetune

        :param net: network object
        :param config: configuration object
        :param chkp_path: path to the checkpoint that needs to be loaded
        :param test_path: path to the folder dedicated to contain the infered files
        :param on_gpu: Train on GPU or CPU
        """
        self.config = config

        # Choose to train on CPU or GPU
        if on_gpu and torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")
        net.to(self.device)

        # Load previous checkpoint
        checkpoint = torch.load(chkp_path, map_location=self.device)
        net.load_state_dict(checkpoint["model_state_dict"])
        self.epoch = checkpoint["epoch"]
        net.eval()
        print("Model and training state restored.")
        self.test_path = test_path

    def classification_test(self, net, test_loader):
        """

        :param net
        :param test_loader
        """
        # Initialize
        # Choose test smoothing parameter (0 for no smothing, 0.99 for big smoothing)
        softmax = torch.nn.Softmax(1)

        # Number of classes predicted by the model
        nc_model = test_loader.dataset.num_classes

        # Initiate global prediction over test clouds
        self.test_probs = np.zeros((test_loader.dataset.num_models, nc_model))
        self.test_counts = np.zeros((test_loader.dataset.num_models, nc_model))

        t = [time.time()]
        mean_dt = np.zeros(1)
        last_display = time.time()
        while np.min(self.test_counts) < self.config["test"]["n_votes"]:

            # Run model on all test examples
            # Initiate result containers
            probs = []
            targets = []
            obj_inds = []

            # Start validation loop
            for batch in test_loader:

                # New time
                t = t[-1:]
                t += [time.time()]

                if "cuda" in self.device.type:
                    batch.to(self.device)

                # Forward pass
                outputs = net(batch, self.config)

                # Get probs and labels
                probs += [softmax(outputs).cpu().detach().numpy()]
                targets += [batch.labels.cpu().numpy()]
                obj_inds += [batch.model_inds.cpu().numpy()]

                if "cuda" in self.device.type:
                    torch.cuda.synchronize(self.device)

                # Average timing
                t += [time.time()]
                mean_dt = 0.95 * mean_dt + 0.05 * (np.array(t[1:]) - np.array(t[:-1]))

                # Display
                if (t[-1] - last_display) > 1.0:
                    last_display = t[-1]
                    message = "Test vote {:.0f} : {:.1f}% (timings : {:4.2f} {:4.2f})"
                    print(
                        message.format(
                            np.min(self.test_counts),
                            100 * len(obj_inds) / self.config["test"]["validation_size"],
                            1000 * (mean_dt[0]),
                            1000 * (mean_dt[1]),
                        )
                    )
            # Stack all validation predictions
            probs = np.vstack(probs)
            targets = np.hstack(targets)
            obj_inds = np.hstack(obj_inds)

            if np.any(test_loader.dataset.input_labels[obj_inds] != targets):
                raise ValueError("wrong object indices")

            # Compute incremental average (predictions are always ordered)
            self.test_counts[obj_inds] += 1
            self.test_probs[obj_inds] += (probs - self.test_probs[obj_inds]) / (
                self.test_counts[obj_inds]
            )

            # Save/Display temporary results
            test_labels = np.array(test_loader.dataset.label_values)

            # Compute classification results
            C1 = fast_confusion(
                test_loader.dataset.input_labels,
                np.argmax(self.test_probs, axis=1),
                test_labels,
            )

            ACC = 100 * np.sum(np.diag(C1)) / (np.sum(C1) + 1e-6)
            print(f"Test Accuracy = {ACC:.1f}%")

    def cloud_segmentation_test(self, net, test_loader):
        """
        Test method for cloud segmentation models

        :param net
        :param test_loader
        """

        # Initialize
        # Choose test smoothing parameter (0 for no smothing, 0.99 for big smoothing)
        test_smooth = 0.95
        test_radius_ratio = 0.7
        softmax = torch.nn.Softmax(1)

        # Number of classes predicted by the model
        nc_model = test_loader.dataset.num_classes

        # Initiate global prediction over test clouds
        self.test_probs = [
            np.zeros((input_label.shape[0], nc_model))
            for input_label in test_loader.dataset.input_labels
        ]

        # Test saving path
        if self.config["model"]["saving"]:
            if not os.path.exists(self.test_path):
                os.makedirs(self.test_path)
            if not os.path.exists(os.path.join(self.test_path, "predictions")):
                os.makedirs(os.path.join(self.test_path, "predictions"))
            if not os.path.exists(os.path.join(self.test_path, "probs")):
                os.makedirs(os.path.join(self.test_path, "probs"))
            if not os.path.exists(os.path.join(self.test_path, "potentials")):
                os.makedirs(os.path.join(self.test_path, "potentials"))

        # If on validation directly compute score
        if test_loader.dataset.task == "validate":
            val_proportions = np.zeros(nc_model, dtype=np.float32)
            i = 0
            for label_value in test_loader.dataset.label_values:
                if label_value not in test_loader.dataset.ignored_labels:
                    val_proportions[i] = np.sum(
                        [
                            np.sum(labels == label_value)
                            for labels in test_loader.dataset.validation_labels
                        ]
                    )
                    i += 1
        else:
            val_proportions = None

        # Network predictions
        test_epoch = 0
        last_saved_min = last_min = -0.5

        t = [time.time()]
        last_display = time.time()
        mean_dt = np.zeros(1)

        # Start test loop
        while True:
            print("Initialize workers")
            for i, batch in enumerate(test_loader):

                # New time
                t = t[-1:]
                t += [time.time()]

                if i == 0:
                    print(f"Done in {t[1] - t[0]:.1f}s")

                if "cuda" in self.device.type:
                    batch.to(self.device)

                # Forward pass
                outputs = net(batch, self.config)

                t += [time.time()]

                # Get probs and labels
                stacked_probs = softmax(outputs).cpu().detach().numpy()
                s_points = batch.points[0].cpu().numpy()
                lengths = batch.lengths[0].cpu().numpy()
                in_inds = batch.input_inds.cpu().numpy()
                cloud_inds = batch.cloud_inds.cpu().numpy()
                if "cuda" in self.device.type:
                    torch.cuda.synchronize(self.device)

                # Get predictions and labels per instance
                i0 = 0
                for b_i, length in enumerate(lengths):

                    # Get prediction
                    points = s_points[i0 : i0 + length]
                    probs = stacked_probs[i0 : i0 + length]
                    inds = in_inds[i0 : i0 + length]
                    c_i = cloud_inds[b_i]

                    if 0 < test_radius_ratio < 1:
                        mask = (
                            np.sum(points**2, axis=1)
                            < (test_radius_ratio * self.config["input"]["sphere_radius"]) ** 2
                        )
                        inds = inds[mask]
                        probs = probs[mask]

                    # Update current probs in whole cloud
                    self.test_probs[c_i][inds] = (
                        test_smooth * self.test_probs[c_i][inds] + (1 - test_smooth) * probs
                    )
                    i0 += length

                # Average timing
                t += [time.time()]
                if i < 2:
                    mean_dt = np.array(t[1:]) - np.array(t[:-1])
                else:
                    mean_dt = 0.9 * mean_dt + 0.1 * (np.array(t[1:]) - np.array(t[:-1]))

                # Display
                if (t[-1] - last_display) > 1.0:
                    last_display = t[-1]
                    message = "e{:03d}-i{:04d} => {:.1f}% (timings : {:4.2f} {:4.2f} {:4.2f})"
                    print(
                        message.format(
                            test_epoch,
                            i,
                            100 * i / self.config["test"]["validation_size"],
                            1000 * (mean_dt[0]),
                            1000 * (mean_dt[1]),
                            1000 * (mean_dt[2]),
                        )
                    )

            # Update minimum od potentials
            new_min = torch.min(test_loader.dataset.min_potentials)
            print(
                "Test epoch {:d}, end. Min potential = {:.2f} (last: {:.1f})".format(
                    test_epoch, new_min, last_min
                )
            )

            # Save predicted cloud
            if last_min + 1 < new_min:
                print("Save predicted cloud...")

                # Update last_min
                last_min += 1

                # Show vote results (On subcloud so it is not the good values here)
                if test_loader.dataset.task == "validate":
                    print("\nConfusion on sub clouds (validation case)")
                    Confs = []
                    for file_idx, _ in enumerate(test_loader.dataset.files):

                        # Insert false columns for ignored labels
                        probs = np.array(self.test_probs[file_idx], copy=True)
                        for l_ind, label_value in enumerate(test_loader.dataset.label_values):
                            if label_value in test_loader.dataset.ignored_labels:
                                probs = np.insert(probs, l_ind, 0, axis=1)

                        # Predicted labels
                        preds = test_loader.dataset.label_values[np.argmax(probs, axis=1)].astype(
                            np.int8
                        )

                        # Targets
                        targets = test_loader.dataset.input_labels[file_idx]

                        # Confs
                        Confs += [fast_confusion(targets, preds, test_loader.dataset.label_values)]

                    # Regroup confusions
                    C = np.sum(np.stack(Confs), axis=0).astype(np.float32)

                    # Remove ignored labels from confusions
                    for l_ind, label_value in reversed(
                        list(enumerate(test_loader.dataset.label_values))
                    ):
                        if label_value in test_loader.dataset.ignored_labels:
                            C = np.delete(C, l_ind, axis=0)
                            C = np.delete(C, l_ind, axis=1)

                    # Rescale with the right number of point per class
                    C *= np.expand_dims(val_proportions / (np.sum(C, axis=1) + 1e-6), 1)

                    # Compute IoUs
                    IoUs = IoU_from_confusions(C)
                    mIoU = np.mean(IoUs)
                    s = f"{100 * mIoU:5.2f} | "
                    for IoU in IoUs:
                        s += f"{100 * IoU:5.2f} "
                    print(s + "\n")

                # Save real IoU once in a while
                if last_saved_min + self.config["test"]["potential_increment"] < new_min:
                    last_saved_min = new_min

                    # Project predictions
                    print(f"\nReproject Vote #{int(np.floor(new_min)):d}")
                    t1 = time.time()
                    self.preds_on_all_files = []
                    for file_idx, file_path in enumerate(test_loader.dataset.files):

                        # The memory requirements for heavy point clouds is too big then we batch
                        # the RAM-consuming structures and write results in several steps
                        proj_gen = test_loader.dataset.generate_projected_point_batches(
                            file_idx, file_path, step=5_000_000
                        )

                        def load_output_generators(
                            file_idx, proj_gen, label_values, ignored_labels
                        ):
                            """Generator function designed to produce batched output structures,
                            for RAM saving purpose.

                            :yield point_batch: xyz coords of point for current batch
                            :yield preds: predicted labels for current batch of points
                            :yield probs: label probabilities for current batch of points
                            """
                            for batch_idx, (point_batch, proj_batch) in enumerate(proj_gen):
                                print(f"Start writing results for batch {batch_idx=}...")
                                # Reproject probs on the evaluations points
                                probs = self.test_probs[file_idx][proj_batch, :]

                                # Insert false columns for ignored labels
                                for l_ind, label_value in enumerate(label_values):
                                    if label_value in ignored_labels:
                                        probs = np.insert(probs, l_ind, 0, axis=1)

                                # Get the predicted labels
                                preds = test_loader.dataset.label_values[
                                    np.argmax(probs, axis=1)
                                ].astype(np.int8)
                                self.preds_on_all_files += [preds]
                                # Deduce the corresponding RGB triplets
                                color_palette = colors.convert_hex_to_rgb(
                                    test_loader.dataset.config["colors"]
                                )
                                yield point_batch, preds, color_palette[preds], probs

                        output_gen = load_output_generators(
                            file_idx,
                            proj_gen,
                            test_loader.dataset.label_values,
                            test_loader.dataset.ignored_labels,
                        )
                        label_names = [
                            "_".join(
                                test_loader.dataset.config["model"]["label_to_names"][label].split()
                            )
                            for label in test_loader.dataset.label_values
                        ]
                        nb_points = test_loader.dataset.test_proj[file_idx].size
                        cloud_name = file_path.name
                        # Save ascii preds
                        ascii_filename = None
                        if test_loader.dataset.task == "test":
                            if test_loader.dataset.config["dataset"].startswith("Semantic3D"):
                                ascii_filename = os.path.join(
                                    self.test_path,
                                    "predictions",
                                    test_loader.dataset.ascii_files[cloud_name],
                                )
                            else:
                                ascii_filename = os.path.join(
                                    self.test_path, "predictions", cloud_name[:-4] + ".txt"
                                )
                        self.write_cloud_segmentation_outputs(
                            cloud_name, output_gen, label_names, nb_points, ascii_filename
                        )

                        # Save potentials
                        pot_points = np.array(
                            test_loader.dataset.pot_trees[file_idx].data, copy=False
                        )
                        pot_filename = os.path.join(self.test_path, "potentials", cloud_name)
                        print(f"Write ply potentials to {pot_filename}...")
                        t1_pot = time.time()
                        pots = test_loader.dataset.potentials[file_idx].numpy().astype(np.float32)
                        write_ply(
                            pot_filename,
                            [pot_points.astype(np.float32), pots],
                            ["x", "y", "z", "pots"],
                        )
                        t2 = time.time()
                        print(f"Done in {t2 - t1_pot:.1f} s\n")

                    t2 = time.time()
                    print(f"Results saved in {t2 - t1:.1f} s\n")

                    # Show vote results
                    if test_loader.dataset.task == "validate":
                        print("Confusion on full clouds")
                        t1 = time.time()
                        Confs = []
                        for file_idx, _ in enumerate(test_loader.dataset.files):

                            # Confusion
                            targets = test_loader.dataset.validation_labels[file_idx]
                            Confs += [
                                fast_confusion(
                                    targets,
                                    self.preds_on_all_files[file_idx],
                                    test_loader.dataset.label_values,
                                )
                            ]

                        t2 = time.time()
                        print(f"Done in {t2 - t1:.1f} s\n")

                        # Regroup confusions
                        C = np.sum(np.stack(Confs), axis=0)

                        # Remove ignored labels from confusions
                        for l_ind, label_value in reversed(
                            list(enumerate(test_loader.dataset.label_values))
                        ):
                            if label_value in test_loader.dataset.ignored_labels:
                                C = np.delete(C, l_ind, axis=0)
                                C = np.delete(C, l_ind, axis=1)

                        IoUs = IoU_from_confusions(C)
                        mIoU = np.mean(IoUs)
                        s = f"{100 * mIoU:5.2f} | "
                        for IoU in IoUs:
                            s += f"{100 * IoU:5.2f} "
                        print("-" * len(s))
                        print(s)
                        print("-" * len(s) + "\n")

            test_epoch += 1

            # Break when reaching number of desired votes
            if last_min > self.config["test"]["n_votes"]:
                print(f"break: {last_min=}, {self.config['test']['n_votes']=}")
                break
            print("---")

    def write_cloud_segmentation_outputs(
        self, cloud_name, output_gen, label_names, nb_points, ascii_filename
    ):
        pred_filename = os.path.join(self.test_path, "predictions", cloud_name)
        if not pred_filename.endswith(".ply"):
            pred_filename += ".ply"
        colorpred_filename = os.path.join(self.test_path, "predictions", "colorized_" + cloud_name)
        if not colorpred_filename.endswith(".ply"):
            colorpred_filename += ".ply"
        prob_filename = os.path.join(self.test_path, "probs", cloud_name)
        if not prob_filename.endswith(".ply"):
            prob_filename += ".ply"
        # First iteration so as to design the header
        points, preds, color_preds, probs = next(output_gen)
        pred_field_list = [points, preds]
        pred_field_names = ["x", "y", "z", "preds"]
        color_pred_field_list = [points, color_preds]
        color_pred_field_names = ["x", "y", "z", "red", "green", "blue"]
        prob_field_list = [points, probs]
        prob_field_names = ["x", "y", "z"] + label_names
        if not ply.check_ply_fields(pred_field_list, pred_field_names):
            print("Invalid field list for predictions. Cancel saving.")
            return
        if not ply.check_ply_fields(color_pred_field_list, color_pred_field_names):
            print("Invalid field list for predictions. Cancel saving.")
            return
        if not ply.check_ply_fields(prob_field_list, prob_field_names):
            print("Invalid field list for predictions. Cancel saving.")
            return
        ply.write_ply_header(pred_filename, pred_field_list, pred_field_names, nb_points=nb_points)
        ply.write_ply_header(
            colorpred_filename, color_pred_field_list, color_pred_field_names, nb_points=nb_points
        )
        ply.write_ply_header(prob_filename, prob_field_list, prob_field_names, nb_points=nb_points)
        ply.write_ply_data(pred_filename, pred_field_list, pred_field_names)
        ply.write_ply_data(colorpred_filename, color_pred_field_list, color_pred_field_names)
        ply.write_ply_data(prob_filename, prob_field_list, prob_field_names)
        if ascii_filename is not None:
            with open(ascii_filename, "wb") as fobj:
                np.savetxt(fobj, preds, fmt="%d")
        # Remaining iterations
        while True:
            try:
                points, preds, color_preds, probs = next(output_gen)
            except StopIteration:
                break
            pred_field_list = [points, preds]
            color_pred_field_list = [points, color_preds]
            prob_field_list = [points, probs]
            if not ply.check_ply_fields(pred_field_list, pred_field_names):
                print("Invalid field list for predictions. Cancel saving.")
                return
            if not ply.check_ply_fields(color_pred_field_list, color_pred_field_names):
                print("Invalid field list for predictions. Cancel saving.")
                return
            if not ply.check_ply_fields(prob_field_list, prob_field_names):
                print("Invalid field list for predictions. Cancel saving.")
                return
            ply.write_ply_data(pred_filename, pred_field_list, pred_field_names)
            ply.write_ply_data(colorpred_filename, color_pred_field_list, color_pred_field_names)
            ply.write_ply_data(prob_filename, prob_field_list, prob_field_names)
            if ascii_filename is not None:
                with open(ascii_filename, "ab") as fobj:
                    np.savetxt(fobj, preds, fmt="%d")

    def slam_segmentation_test(self, net, test_loader, debug=True):
        """
        Test method for slam segmentation models

        :param net:
        :param test_loader:
        :param debug:
        """

        ############
        # Initialize
        ############

        # Choose validation smoothing parameter (0 for no smothing, 0.99 for big smoothing)
        test_smooth = 0.5
        last_min = -0.5
        softmax = torch.nn.Softmax(1)

        # Number of classes including ignored labels
        nc_tot = test_loader.dataset.num_classes
        nc_model = net.C

        # Test saving path
        report_path = None
        if self.config["model"]["saving"]:
            if not os.path.exists(self.test_path):
                os.makedirs(self.test_path)
            report_path = os.path.join(self.test_path, "reports")
            if not os.path.exists(report_path):
                os.makedirs(report_path)

        if test_loader.dataset.task == "validate":
            for folder in ["val_predictions", "val_probs"]:
                if not os.path.exists(os.path.join(self.test_path, folder)):
                    os.makedirs(os.path.join(self.test_path, folder))
        else:
            for folder in ["predictions", "probs"]:
                if not os.path.exists(os.path.join(self.test_path, folder)):
                    os.makedirs(os.path.join(self.test_path, folder))

        # Init validation container
        all_f_preds = []
        all_f_labels = []
        if test_loader.dataset.task == "validate":
            for seq_frames in test_loader.dataset.frames:
                all_f_preds.append([np.zeros((0,), dtype=np.int32) for _ in seq_frames])
                all_f_labels.append([np.zeros((0,), dtype=np.int32) for _ in seq_frames])

        # Network predictions
        predictions = []
        targets = []
        test_epoch = 0

        t = [time.time()]
        last_display = time.time()
        mean_dt = np.zeros(1)

        # Start test loop
        while True:
            print("Initialize workers")
            for i, batch in enumerate(test_loader):

                # New time
                t = t[-1:]
                t += [time.time()]

                if i == 0:
                    print(f"Done in {t[1] - t[0]:.1f}s")

                if "cuda" in self.device.type:
                    batch.to(self.device)

                # Forward pass
                outputs = net(batch, self.config)

                # Get probs and labels
                stk_probs = softmax(outputs).cpu().detach().numpy()
                lengths = batch.lengths[0].cpu().numpy()
                f_inds = batch.frame_inds.cpu().numpy()
                r_inds_list = batch.reproj_inds
                r_mask_list = batch.reproj_masks
                labels_list = batch.val_labels
                if "cuda" in self.device.type:
                    torch.cuda.synchronize(self.device)

                t += [time.time()]

                # Get predictions and labels per instance
                i0 = 0
                for b_i, length in enumerate(lengths):

                    # Get prediction
                    probs = stk_probs[i0 : i0 + length]
                    proj_inds = r_inds_list[b_i]
                    proj_mask = r_mask_list[b_i]
                    frame_labels = labels_list[b_i]
                    s_ind = f_inds[b_i, 0]
                    f_ind = f_inds[b_i, 1]

                    # Project predictions on the frame points
                    proj_probs = probs[proj_inds]

                    # Safe check if only one point:
                    if proj_probs.ndim < 2:
                        proj_probs = np.expand_dims(proj_probs, 0)

                    # Save probs in a binary file (uint8 format for lighter weight)
                    seq_name = test_loader.dataset.sequences[s_ind]
                    if test_loader.dataset.task == "validate":
                        folder = "val_probs"
                        pred_folder = "val_predictions"
                    else:
                        folder = "probs"
                        pred_folder = "predictions"
                    filename = f"{seq_name}_{f_ind:7d}.npy"
                    filepath = os.path.join(self.test_path, folder, filename)
                    if os.path.exists(filepath):
                        frame_probs_uint8 = np.load(filepath)
                    else:
                        frame_probs_uint8 = np.zeros((proj_mask.shape[0], nc_model), dtype=np.uint8)
                    frame_probs = frame_probs_uint8[proj_mask, :].astype(np.float32) / 255
                    frame_probs = test_smooth * frame_probs + (1 - test_smooth) * proj_probs
                    frame_probs_uint8[proj_mask, :] = (frame_probs * 255).astype(np.uint8)
                    np.save(filepath, frame_probs_uint8)

                    # Save some prediction in ply format for visual
                    if test_loader.dataset.task == "validate":

                        # Insert false columns for ignored labels
                        frame_probs_uint8_bis = frame_probs_uint8.copy()
                        for l_ind, label_value in enumerate(test_loader.dataset.label_values):
                            if label_value in test_loader.dataset.ignored_labels:
                                frame_probs_uint8_bis = np.insert(
                                    frame_probs_uint8_bis, l_ind, 0, axis=1
                                )

                        # Predicted labels
                        frame_preds = test_loader.dataset.label_values[
                            np.argmax(frame_probs_uint8_bis, axis=1)
                        ].astype(np.int32)

                        # Save some of the frame pots
                        if f_ind % 20 == 0:
                            seq_path = os.path.join(
                                test_loader.dataset.path,
                                "sequences",
                                test_loader.dataset.sequences[s_ind],
                            )
                            velo_file = os.path.join(
                                seq_path,
                                "velodyne",
                                test_loader.dataset.frames[s_ind][f_ind] + ".bin",
                            )
                            frame_points = np.fromfile(velo_file, dtype=np.float32)
                            frame_points = frame_points.reshape((-1, 4))
                            predpath = os.path.join(
                                self.test_path, pred_folder, filename[:-4] + ".ply"
                            )
                            pots = np.zeros((0,))  # test_loader.dataset.f_potentials[s_ind][f_ind]
                            if pots.shape[0] > 0:
                                write_ply(
                                    predpath,
                                    [
                                        frame_points[:, :3],
                                        frame_labels,
                                        frame_preds,
                                        pots,
                                    ],
                                    ["x", "y", "z", "gt", "pre", "pots"],
                                )
                            else:
                                write_ply(
                                    predpath,
                                    [frame_points[:, :3], frame_labels, frame_preds],
                                    ["x", "y", "z", "gt", "pre"],
                                )

                            # Also Save lbl probabilities
                            probpath = os.path.join(
                                self.test_path, folder, filename[:-4] + "_probs.ply"
                            )
                            lbl_names = [
                                test_loader.dataset.config["model"]["label_to_names"][label_value]
                                for label_value in test_loader.dataset.label_values
                                if label_value not in test_loader.dataset.ignored_labels
                            ]
                            write_ply(
                                probpath,
                                [frame_points[:, :3], frame_probs_uint8],
                                ["x", "y", "z"] + lbl_names,
                            )

                        # keep frame preds in memory
                        all_f_preds[s_ind][f_ind] = frame_preds
                        all_f_labels[s_ind][f_ind] = frame_labels

                    else:

                        # Save some of the frame preds
                        if f_inds[b_i, 1] % 100 == 0:

                            # Insert false columns for ignored labels
                            for l_ind, label_value in enumerate(test_loader.dataset.label_values):
                                if label_value in test_loader.dataset.ignored_labels:
                                    frame_probs_uint8 = np.insert(
                                        frame_probs_uint8, l_ind, 0, axis=1
                                    )

                            # Predicted labels
                            frame_preds = test_loader.dataset.label_values[
                                np.argmax(frame_probs_uint8, axis=1)
                            ].astype(np.int32)

                            # Load points
                            seq_path = os.path.join(
                                test_loader.dataset.path,
                                "sequences",
                                test_loader.dataset.sequences[s_ind],
                            )
                            velo_file = os.path.join(
                                seq_path,
                                "velodyne",
                                test_loader.dataset.frames[s_ind][f_ind] + ".bin",
                            )
                            frame_points = np.fromfile(velo_file, dtype=np.float32)
                            frame_points = frame_points.reshape((-1, 4))
                            predpath = os.path.join(
                                self.test_path, pred_folder, filename[:-4] + ".ply"
                            )
                            pots = np.zeros((0,))  # test_loader.dataset.f_potentials[s_ind][f_ind]
                            if pots.shape[0] > 0:
                                write_ply(
                                    predpath,
                                    [frame_points[:, :3], frame_preds, pots],
                                    ["x", "y", "z", "pre", "pots"],
                                )
                            else:
                                write_ply(
                                    predpath,
                                    [frame_points[:, :3], frame_preds],
                                    ["x", "y", "z", "pre"],
                                )

                    # Stack all prediction for this epoch
                    i0 += length

                # Average timing
                t += [time.time()]
                mean_dt = 0.95 * mean_dt + 0.05 * (np.array(t[1:]) - np.array(t[:-1]))

                # Display
                if (t[-1] - last_display) > 1.0:
                    last_display = t[-1]
                    message = (
                        "e{:03d}-i{:04d} => {:.1f}% "
                        "(timings : {:4.2f} {:4.2f} {:4.2f}) / pots {:d} => {:.1f}%"
                    )
                    min_pot = int(torch.floor(torch.min(test_loader.dataset.potentials)))
                    pot_num = (
                        torch.sum(test_loader.dataset.potentials > min_pot + 0.5)
                        .type(torch.int32)
                        .item()
                    )
                    current_num = (
                        pot_num
                        + (i + 1 - self.config["test"]["validation_size"])
                        * self.config["test"]["val_batch_num"]
                    )
                    print(
                        message.format(
                            test_epoch,
                            i,
                            100 * i / self.config["test"]["validation_size"],
                            1000 * (mean_dt[0]),
                            1000 * (mean_dt[1]),
                            1000 * (mean_dt[2]),
                            min_pot,
                            100.0 * current_num / len(test_loader.dataset.potentials),
                        )
                    )

            # Update minimum od potentials
            new_min = torch.min(test_loader.dataset.potentials)
            print(f"Test epoch {test_epoch:d}, end. Min potential = {new_min:.1f}")

            if last_min + 1 < new_min:

                # Update last_min
                last_min += 1

                if test_loader.dataset.task == "validate" and last_min % 1 == 0:

                    # Results on the whole validation set
                    # Confusions for our subparts of validation set
                    Confs = np.zeros((len(predictions), nc_tot, nc_tot), dtype=np.int32)
                    for i, (preds, truth) in enumerate(zip(predictions, targets)):

                        # Confusions
                        Confs[i, :, :] = fast_confusion(
                            truth, preds, test_loader.dataset.label_values
                        ).astype(np.int32)

                    # Show vote results
                    print("\nCompute confusion")

                    val_preds = []
                    val_labels = []
                    t1 = time.time()
                    for frame_idx, _ in enumerate(test_loader.dataset.frames):
                        val_preds += [np.hstack(all_f_preds[frame_idx])]
                        val_labels += [np.hstack(all_f_labels[frame_idx])]
                    val_preds = np.hstack(val_preds)
                    val_labels = np.hstack(val_labels)
                    t2 = time.time()
                    C_tot = fast_confusion(val_labels, val_preds, test_loader.dataset.label_values)
                    t3 = time.time()
                    print(f" Stacking time : {t2 - t1:.1f}s")
                    print(f"Confusion time : {t3 - t2:.1f}s")

                    s1 = "\n"
                    for cc in C_tot:
                        for c in cc:
                            s1 += f"{c:7.0f} "
                        s1 += "\n"
                    if debug:
                        print(s1)

                    # Remove ignored labels from confusions
                    for l_ind, label_value in reversed(
                        list(enumerate(test_loader.dataset.label_values))
                    ):
                        if label_value in test_loader.dataset.ignored_labels:
                            C_tot = np.delete(C_tot, l_ind, axis=0)
                            C_tot = np.delete(C_tot, l_ind, axis=1)

                    # Objects IoU
                    val_IoUs = IoU_from_confusions(C_tot)

                    # Compute IoUs
                    mIoU = np.mean(val_IoUs)
                    s2 = f"{100 * mIoU:5.2f} | "
                    for IoU in val_IoUs:
                        s2 += f"{100 * IoU:5.2f} "
                    print(s2 + "\n")

                    # Save a report
                    report_file = os.path.join(
                        report_path, f"report_{int(np.floor(last_min)):04d}.txt"
                    )
                    str_report = "Report of the confusion and metrics\n"
                    str_report += "***********************************\n\n\n"
                    str_report += "Confusion matrix:\n\n"
                    str_report += s1
                    str_report += "\nIoU values:\n\n"
                    str_report += s2
                    str_report += "\n\n"
                    with open(report_file, "w") as f:
                        f.write(str_report)

            test_epoch += 1

            # Break when reaching number of desired votes
            if last_min > self.config["test"]["n_votes"]:
                break
