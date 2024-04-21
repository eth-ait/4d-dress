import numpy as np

# manually assign mask_labels
def update_manual_mask_labels(frame, label, masks, manual_mask_labels):
    # cat manual_masks as [[view, mask, label], ]
    manual_masks = np.concatenate([np.stack(masks, axis=0), np.ones((len(masks), 1)) * label], axis=1).astype(int)
    # append manual_masks to manual_mask_labels
    manual_mask_labels[frame] = np.concatenate([manual_mask_labels[frame], manual_masks], axis=0) if frame in manual_mask_labels else manual_masks
    return manual_mask_labels


# manually assign mask labels
def update_manual_labels(subj='', outfit='', seq='', surface_labels=None):
    # init manual_labels {}
    manual_labels = dict()

    # # assign manual_labels for subj_outfit_seq: skin0, hair1, shoe2, upper3, lower4, outer5
    # if subj=='00122' and outfit == 'Outer' and seq == 'Take9':
    #     # init manual_mask_labels: {'frame': [[view, mask, label], ...]}
    #     manual_labels['mask_labels'] = dict()
    #     # update manual_mask_labels
    #     frame, label = '00040', surface_labels.index('upper')  # locate target frame and target label
    #     masks = [[0, 13], [18, 12]]  # locate manual region as [[view_id, mask_id], ...]
    #     update_manual_mask_labels(frame, label, masks, manual_labels['mask_labels'])
    #     # update manual_mask_labels
    #     frame, label = '00041', surface_labels.index('upper')
    #     masks = [[18, 12]]
    #     update_manual_mask_labels(frame, label, masks, manual_labels['mask_labels'])
    #     # update manual_mask_labels
    #     frame, label = '00042', surface_labels.index('upper')
    #     masks = [[18, 13]]
    #     update_manual_mask_labels(frame, label, masks, manual_labels['mask_labels'])
    #     # update manual_mask_labels
    #     frame, label = '00043', surface_labels.index('upper')
    #     masks = [[18, 11]]
    #     update_manual_mask_labels(frame, label, masks, manual_labels['mask_labels'])
    #     # update manual_mask_labels
    #     frame, label = '00044', surface_labels.index('upper')
    #     masks = [[18, 14]]
    #     update_manual_mask_labels(frame, label, masks, manual_labels['mask_labels'])


    # # assign manual_labels for subj_outfit_seq: skin0, hair1, shoe2, upper3, lower4, outer5
    # if subj=='00129' and outfit == 'Outer' and seq == 'Take9':
    #     # init manual_mask_labels: {'frame': [[view, mask, label], ...]}
    #     manual_labels['mask_labels'] = dict()
    #     # update manual_mask_labels
    #     frame, label = '00001', surface_labels.index('skin')
    #     masks = [[4, 18]]
    #     update_manual_mask_labels(frame, label, masks, manual_labels['mask_labels'])
    #     # update manual_mask_labels
    #     frame, label = '00001', surface_labels.index('hair')
    #     masks = [[8, 7], [9, 8], [10, 8], [11, 7]]
    #     update_manual_mask_labels(frame, label, masks, manual_labels['mask_labels'])
    #     # update manual_mask_labels
    #     frame, label = '00001', surface_labels.index('upper')
    #     masks = [[3, 7], [4, 8], [5, 8]]
    #     update_manual_mask_labels(frame, label, masks, manual_labels['mask_labels'])
    #     # update manual_mask_labels
    #     frame, label = '00001', surface_labels.index('outer')
    #     masks = [[3, 2], [4, 2], [5, 2], [9, 2], [10, 2], [11, 2], [17, 2], [20, 2]]
    #     update_manual_mask_labels(frame, label, masks, manual_labels['mask_labels'])


    # assign manual_labels for subj_outfit_seq: skin0, hair1, shoe2, upper3, lower4, outer5
    if subj=='00135' and outfit == 'Inner' and seq == 'Take1':
        # init manual_mask_labels: {'frame': [[view, mask, label], ...]}
        manual_labels['mask_labels'] = dict()
        
        # update manual_effort for new_label
        if 'sock' in surface_labels:
            # update manual_mask_labels
            frame, label = '00006', surface_labels.index('sock')
            masks = [[1, 15], [1, 17], [3, 16], [3, 17], [5, 11], [7, 15], [7, 16], [10, 16], [10, 17], [11, 9]]
            update_manual_mask_labels(frame, label, masks, manual_labels['mask_labels'])
            # update manual_mask_labels
            frame, label = '00006', surface_labels.index('shoe')
            masks = [[1, 10], [1, 14], [3, 12], [3, 15], [5, 8], [7, 12], [7, 13], [10, 12], [10, 14], [11, 6]]
            update_manual_mask_labels(frame, label, masks, manual_labels['mask_labels'])

    # return manual_efforts = {'manual_labels'}
    return {'manual_labels': manual_labels}
