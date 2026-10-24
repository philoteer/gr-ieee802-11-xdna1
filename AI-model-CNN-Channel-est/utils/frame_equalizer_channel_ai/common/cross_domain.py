"""Shared cross-domain policy: select on source validation, test on target only."""

SOURCE_TRAIN_PARTITIONS = ('train', 'test')
TARGET_TEST_PARTITIONS = ('train', 'val', 'test')


def domain_policy(mode):
    source, target = mode.split('_to_')
    return {
        'policy': 'source_train_test_val_target_test',
        'partition_version': 'by_frame', 'source': source, 'target': target,
        'train': {'domain': source, 'partitions': list(SOURCE_TRAIN_PARTITIONS)},
        'validation': {'domain': source, 'partitions': ['val']},
        'test': {'domain': target, 'partitions': list(TARGET_TEST_PARTITIONS)},
        'selection_domain': source, 'test_is_validation': False,
    }


def source_frames(partitions):
    return {'train': [frame for name in SOURCE_TRAIN_PARTITIONS for frame in partitions[name]],
            'val': list(partitions['val'])}


def target_frames(partitions):
    return [frame for name in TARGET_TEST_PARTITIONS for frame in partitions[name]]
