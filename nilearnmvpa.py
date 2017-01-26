def nilearnmvpa(training_filename, testing_filename, anatomical_filename, condition_label_filename, mask_filename = None):
    # Abigail Novick 1/25/2017
    # Example mvpa function, takes fmri data to train and test. Anatomical image is optional, but nice for plotting.
    # You need to have condition labels for your TRs for your training period.
    # Masking is optional.


    import nilearn
    import numpy as np
    import scipy.io as sio

    #first: import nii files into python

    fmri_filename = training_filename
    dnms_filename = testing_filename
    anat_filename = anatomical_filename
    matlabLabels = sio.loadmat(condition_label_filename)
    labelArray = matlabLabels['StimuliMatrix']

    from nilearn.input_data import NiftiMasker
    masker = NiftiMasker(mask_img=mask_filename, standardize=True)

    # We give the masker a filename and retrieve a 2D array ready
    # for machine learning with scikit-learn
    fmri_masked = masker.fit_transform(fmri_filename)
    dnms_masked = masker.fit_transform(dnms_filename)

    #good to check the training set fmri_masked and test set dnms_masked have the same number of voxels
    #data format = TRs, voxels

    # Load TR label information. key for training the classifier!

    # need to turn the 4*numTrs array into a vector of strings
    # rests = 0, leftface = 1, rightface = 2, leftscene = 3, rightscene = 4
    maxes = np.max(labelArray, axis=0)
    rest_mask = maxes == 0
    labelVector_numbers = np.argmax(labelArray, axis=0) + 1
    labelVector_numbers[rest_mask] = 0
    labelVector = labelVector_numbers.astype(str)
    # this might be a good place in your code to account for the hemodynamic lag. up to you though!

    condition_mask = labelVector != "0"

    #take out rests from training set
    fmri_masked_norest = fmri_masked[condition_mask]
    print(fmri_masked_norest.shape)
    condition_norest = labelVector[condition_mask]
    print(condition_norest.shape)


    # time to train the svc
    # NOTE: this is trained and tested on the same data set. so it's useless. just proof of concept.
    from sklearn.svm import SVC
    svc = SVC(kernel='linear', probability = True)
    print(svc)

    svc.fit(fmri_masked_norest, condition_norest)

    #time to test predictions for the training and test set
    prediction = svc.predict(fmri_masked_norest)
    dnms_prediction = svc.predict_log_proba(dnms_masked)
    print(dnms_prediction)

    print((prediction == condition_norest).sum() / float(len(condition_norest)))


    #here we leave 30 TRs out for testing. much better idea!

    import sklearn.svm
    import sklearn.linear_model
    svc_split = sklearn.linear_model.LogisticRegression(multi_class='multinomial', solver='newton-cg')
    #svc_split = sklearn.svm.NuSVC(kernel='linear')

    svc_split.fit(fmri_masked_norest[:-30,:], condition_norest[:-30])

    prediction = svc_split.predict(fmri_masked_norest[-30:])
    print((prediction == condition_norest[-30:]).sum() / float(len(condition_norest[-30:])))


    #even better than that, cross validation!!!
    from sklearn.cross_validation import KFold

    cv = KFold(n=len(fmri_masked_norest), n_folds=5)
    target=condition_norest
    for train, test in cv:
        svc_split.fit(fmri_masked_norest[train], target[train])
        prediction_split = svc_split.predict(fmri_masked_norest[test])
        print((prediction_split == target[test]).sum() / float(len(target[test])))


    from sklearn.cross_validation import cross_val_score
    cv_score = cross_val_score(svc_split, fmri_masked_norest, target)
    print(cv_score)


    # i don't know what the cv=cv score flag does....
    cv_score = cross_val_score(svc_split, fmri_masked_norest, target, cv=cv)
    print(cv_score)

    # # here i test the classifier on new test data. I will use this later to correlate classifier evidence with RTs
    # # Using predict_log_proba outputs a log probability score for every classifier for every TR
    dnms_prediction_split = svc_split.predict_log_proba(dnms_masked)
    classifier_scores = dnms_prediction_split

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #the rest of this commented out bit is for plotting your coefficient weights
    # simple_coefs = svc_split.coef_
    #
    # print(simple_coefs.shape)
    # coef0 = simple_coefs[0, :]
    # coef1 = simple_coefs[1, :]
    # coef2 = simple_coefs[2, :]
    # coef3 = simple_coefs[3, :]
    #
    # coef_img0 = masker.inverse_transform(coef0)
    # coef_img1 = masker.inverse_transform(coef1)
    # coef_img2 = masker.inverse_transform(coef2)
    # coef_img3 = masker.inverse_transform(coef3)
    #
    #
    # coef_img0.to_filename('inception0_svc_weights.nii.gz')
    # coef_img1.to_filename('inception1_svc_weights.nii.gz')
    # coef_img2.to_filename('inception2_svc_weights.nii.gz')
    # coef_img3.to_filename('inception3_svc_weights.nii.gz')

    #I have 4 classifiers, so that's why I do each of these steps 4 times.


    # # In[49]:
    #
    # from nilearn.plotting import plot_stat_map, show
    #
    # plot_stat_map(coef_img0, anat_filename,
    #               title="SVM weights", display_mode="yx")
    #
    # show()
    #
    #
    # # In[21]:
    #
    # from nilearn.plotting import plot_stat_map, show
    #
    # plot_stat_map(coef_img1, anat_filename,
    #               title="SVM weights", display_mode="yx")
    #
    # show()
    #
    #
    # # In[22]:
    #
    # from nilearn.plotting import plot_stat_map, show
    #
    # plot_stat_map(coef_img2, anat_filename,
    #               title="SVM weights", display_mode="yx")
    #
    # show()
    #
    #
    # # In[23]:
    #
    # from nilearn.plotting import plot_stat_map, show
    #
    # plot_stat_map(coef_img3, anat_filename,
    #               title="SVM weights", display_mode="yx")
    #
    # show()


    return classifier_scores


