from datasets.kinetics import Kinetics
from datasets.activitynet import ActivityNet
from datasets.ucf101 import UCF101
from datasets.ucf101_mv_residuals import UCF101 as UCF101R
from datasets.ucf_mult_loss import UCF101 as UCF101M

from datasets.hmdb51 import HMDB51
from datasets.hmdb51_mult_loss import HMDB51 as HMDB51M
from datasets.hmdb51_mv_residuals import HMDB51 as HMDB51R


def get_training_set(opt, spatial_transform, temporal_transform,
                     target_transform):
    assert opt.dataset in ['kinetics', 'activitynet', 'ucf101', 'hmdb51']

    if opt.dataset == 'kinetics':
        training_data = Kinetics(opt,
            opt.video_path,
            opt.annotation_path,
            'training',
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform,
            target_transform=target_transform)
    elif opt.dataset == 'activitynet':
        training_data = ActivityNet(
            opt.video_path,
            opt.annotation_path,
            'training',
            False,
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform,
            target_transform=target_transform)
    elif opt.dataset == 'ucf101':
        if opt.compressed:
            if opt.mult_loss:
                training_data = UCF101M(
                        opt,
                        opt.annotation_path,
                        'training',
                        spatial_transform=spatial_transform,
                        temporal_transform=temporal_transform,
                        target_transform=target_transform)
            else:
                training_data = UCF101R(    
                    opt,
                    opt.annotation_path,
                    'training',
                    spatial_transform=spatial_transform,
                    temporal_transform=temporal_transform,
                    target_transform=target_transform)
        else:
            training_data = UCF101(opt,
                opt.jpeg_path,
                opt.annotation_path,
                'training',
                spatial_transform=spatial_transform,
                temporal_transform=temporal_transform,
                target_transform=target_transform)
    elif opt.dataset == 'hmdb51':
        if opt.compressed:
            if opt.mult_loss:
                training_data = HMDB51M(
                    opt,
                    opt.annotation_path,
                    'training',
                    spatial_transform=spatial_transform,
                    temporal_transform=temporal_transform,
                    target_transform=target_transform)
            else:
                training_data = HMDB51R(
                    opt,
                    opt.annotation_path,
                    'training',
                    0,
                    spatial_transform=spatial_transform,
                    temporal_transform=temporal_transform,
                    target_transform=target_transform)
        else:
            training_data = HMDB51(opt,
                opt.jpeg_path,
                opt.annotation_path,
                'training',
                spatial_transform=spatial_transform,
                temporal_transform=temporal_transform,
                target_transform=target_transform)

    return training_data


def get_validation_set(opt, spatial_transform, temporal_transform,
                       target_transform=None):
    assert opt.dataset in ['kinetics', 'activitynet', 'ucf101', 'hmdb51']
    if opt.video_level_accuracy:
        val_or_test = 'test'
        if opt.dataset == 'kinetics' or  opt.dataset == 'ucf101':
            val_or_test = 'validation'

        n_val_samples = 0
    else:
        val_or_test = 'validation'
        n_val_samples = opt.n_val_samples
    
    if opt.dataset == 'kinetics':
        validation_data = Kinetics(opt,
            opt.video_path,
            opt.annotation_path,
            val_or_test,
            n_val_samples,
            spatial_transform,
            temporal_transform,
            target_transform,
            frames_sequence=opt.sample_duration)
    elif opt.dataset == 'activitynet':
        validation_data = ActivityNet(
            opt.jpeg_path,
            opt.annotation_path,
            val_or_test,
            False,
            opt.n_val_samples,
            spatial_transform,
            temporal_transform,
            target_transform,
            sample_duration=opt.sample_duration)
    elif opt.dataset == 'ucf101':
        if opt.compressed:
            validation_data = UCF101R(
                opt,
                opt.annotation_path,
                val_or_test,
                n_val_samples,
                spatial_transform=spatial_transform,
                temporal_transform=temporal_transform,
                target_transform=target_transform)
            
                
        else:
            validation_data = UCF101(opt,
                opt.jpeg_path,
                opt.annotation_path,
                val_or_test,
                n_val_samples,
                spatial_transform=spatial_transform,
                temporal_transform=temporal_transform,
                target_transform=target_transform,
                sample_duration=opt.sample_duration)

    elif opt.dataset == 'hmdb51':
        if opt.compressed:
            validation_data = HMDB51R(
                opt,
                opt.annotation_path,
                val_or_test,
                n_val_samples,
                spatial_transform=spatial_transform,
                temporal_transform=temporal_transform,
                target_transform=target_transform,
                sample_duration=opt.sample_duration)
            
        else:
            validation_data = HMDB51(opt,
                opt.jpeg_path,
                opt.annotation_path,
                val_or_test,
                n_val_samples,
                spatial_transform,
                temporal_transform,
                target_transform,
                sample_duration=opt.sample_duration)
    return validation_data

"""
def get_test_set(opt, spatial_transform, temporal_transform, target_transform):
    assert opt.dataset in ['kinetics', 'activitynet', 'ucf101', 'hmdb51']
    assert opt.test_subset in ['val', 'test']


        subset = 'testing'
    if opt.dataset == 'kinetics':
        test_data = Kinetics(
            opt.video_path,
            opt.annotation_path,
            subset,
            0,
            spatial_transform,
            temporal_transform,
            target_transform,
            sample_duration=opt.sample_duration)
    elif opt.dataset == 'activitynet':
        test_data = ActivityNet(
            opt.jpeg_path,
            opt.annotation_path,
            subset,
            True,
            0,
            spatial_transform,
            temporal_transform,
            target_transform,
            sample_duration=opt.sample_duration)
    elif opt.dataset == 'ucf101':
        test_data = UCF101(
            opt.jpeg_path,
            opt.annotation_path,
            subset,
            0,
            spatial_transform,
            temporal_transform,
            target_transform,
            sample_duration=opt.sample_duration)
    elif opt.dataset == 'hmdb51':
        test_data = HMDB51(
            opt.jpeg_path,
            opt.annotation_path,
            subset,
            0,
            spatial_transform,
            temporal_transform,
            target_transform,
            sample_duration=opt.sample_duration)

    return test_data
"""