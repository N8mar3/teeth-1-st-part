import torch
import torch.optim as opt
from _Unet_architecture import UNet
from _data_builder import DataBuilder
from _data_loader import get_loaders
from _trainer import TrainFunction
from _work_utils import WorkFunction


def train_model(
                        data_path,
                        data_num_classes,
                        data_list_of_categories,
                        data_format,
                        data_normalization,
                        data_fit,
                        data_save,
                        mask_path,
                        mask_num_classes,
                        mask_list_of_categories,
                        mask_format,
                        mask_normalization,
                        mask_fit,
                        mask_save,
                        model_name,
                        model_name_to_load,
                        model_input_class,
                        model_output_class,
                        model_num_of_epochs,
                        model_learning_rate,
                        model_number_of_unet_blocks,
                        model_number_of_start_filters,
                        model_activation_function,
                        model_normalization,
                        dims,
                        model_convolution_mode,
                        model_loss_function_mode_binary,
                        model_transfer_learning,
                        num_of_chunks,
                        num_of_workers,
                        batch_size,
                        augmentation_coefficient
                ):
    device = torch.device('cpu')
    scale = torch.cuda.amp.GradScaler()
    optimizer = opt.RMSprop
    data = DataBuilder(
                        data_path=data_path,
                        data_format=data_format,
                        list_of_categories=data_list_of_categories,
                        num_of_chunks=num_of_chunks,
                        augmentation_coeff=augmentation_coefficient,
                        num_of_classes=data_num_classes,
                        normalise=data_normalization,
                        fit=data_fit,
                        save_data=data_save,
                        ).forward()

    mask = DataBuilder(
                        data_path=mask_path,
                        data_format=mask_format,
                        list_of_categories=mask_list_of_categories,
                        num_of_chunks=num_of_chunks,
                        augmentation_coeff=augmentation_coefficient,
                        num_of_classes=mask_num_classes,
                        normalise=mask_normalization,
                        fit=mask_fit,
                        save_data=mask_save,
                        ).forward()

    loader = get_loaders(
                        data, mask,
                        batch_size=batch_size,
                        num_workers=num_of_workers
                        )

    model = UNet(
                        in_channels=model_input_class,          out_channels=model_output_class,
                        n_blocks=model_number_of_unet_blocks,   start_filters=model_number_of_start_filters,
                        activation=model_activation_function,   normalization=model_normalization,
                        conv_mode=model_convolution_mode,       dim=dims
                        ).to(device)

    TrainFunction(
                        data_loader=loader,
                        device_for_training=device,
                        model_name=model_name,
                        model_name_pretrained=model_name_to_load,
                        model=model,
                        optimizer=optimizer,
                        learning_rate=model_learning_rate,
                        scale=scale,
                        num_epochs=model_num_of_epochs,
                        transfer_learning=model_transfer_learning,
                        binary_loss_f=model_loss_function_mode_binary
                        ).forward()


def work(
                        data_1_path,
                        data_1_list_of_categories,
                        data_1_number_of_classes_to_split,
                        data_1_data_format,
                        data_1_normalization,
                        data_1_fit,
                        data_1_save,
                        model_1_name_to_load,
                        model_1_input_class,
                        model_1_output_class,
                        model_1_number_of_unet_blocks,
                        model_1_number_of_start_filters,
                        model_1_activation_function,
                        model_1_normalization,
                        model_1_dims,
                        model_1_convolution_mode,
                        work_1_save_3d_object,
                        work_1_show_3d_object,
                        work_1_stage_two,
                        work_1_smooth,
                        data_2_list_of_categories,
                        data_2_number_of_classes_to_split,
                        data_2_data_format,
                        data_2_normalization,
                        data_2_fit,
                        data_2_save,
                        model_2_name_to_load,
                        model_2_input_class,
                        model_2_output_class,
                        model_2_number_of_unet_blocks,
                        model_2_number_of_start_filters,
                        model_2_activation_function,
                        model_2_normalization,
                        model_2_dims,
                        model_2_convolution_mode,
                        work_2_save_3d_object,
                        work_2_show_3d_object,
                        work_2_stage_two,
                        work_2_smooth,
                        num_of_chunks,
                        num_workers_wrk,
                        batch_size_wrk,
                        augmentation_coefficient
             ):
    device = torch.device("cuda:0" if True else "cpu")

    data_1 = DataBuilder(
                        data_path=data_1_path,
                        data_format=data_1_data_format,
                        list_of_categories=data_1_list_of_categories,
                        num_of_chunks=num_of_chunks,
                        augmentation_coeff=augmentation_coefficient,
                        num_of_classes=data_1_number_of_classes_to_split,
                        normalise=data_1_normalization,
                        fit=data_1_fit,
                        save_data=data_1_save,
                        ).forward()

    mask_1 = mask_2 = None

    loader_1 = get_loaders(
                        data_1, mask_1,
                        batch_size=batch_size_wrk,
                        num_workers=num_workers_wrk
    )
    model_1 = UNet(
                        in_channels=model_1_input_class,          out_channels=model_1_output_class,
                        n_blocks=model_1_number_of_unet_blocks,   start_filters=model_1_number_of_start_filters,
                        activation=model_1_activation_function,   normalization=model_1_normalization,
                        conv_mode=model_1_convolution_mode,       dim=model_1_dims
                        ).to(device)

    work_1 = WorkFunction(
                        data_loader=loader_1,
                        device_for_training=device,
                        model_name=model_1_name_to_load,
                        model=model_1,
                        save_3d=work_1_save_3d_object,
                        show_prediction=work_1_show_3d_object,
                        stage=work_1_stage_two,
                        smooth=work_1_smooth
                        ).predictor()

    data_2 = DataBuilder(
                        data_path=work_1,
                        data_format=data_2_data_format,
                        list_of_categories=data_2_list_of_categories,
                        num_of_chunks=num_of_chunks,
                        augmentation_coeff=augmentation_coefficient,
                        num_of_classes=data_2_number_of_classes_to_split,
                        normalise=data_2_normalization,
                        fit=data_2_fit,
                        save_data=data_2_save,
                        ).forward()

    loader_2 = get_loaders(
                        data_2, mask_2,
                        batch_size=batch_size_wrk,
                        num_workers=num_workers_wrk
    )
    model_2 = UNet(
                        in_channels=model_2_input_class,        out_channels=model_2_output_class,
                        n_blocks=model_2_number_of_unet_blocks, start_filters=model_2_number_of_start_filters,
                        activation=model_2_activation_function, normalization=model_2_normalization,
                        conv_mode=model_2_convolution_mode,     dim=model_2_dims
                        ).to(device)

    WorkFunction(
                        data_loader=loader_2,
                        device_for_training=device,
                        model_name=model_2_name_to_load,
                        model=model_2,
                        save_3d=work_2_save_3d_object,
                        show_prediction=work_2_show_3d_object,
                        stage=work_2_stage_two,
                        smooth=work_2_smooth
                        ).predictor()
