
# coding: utf-8

# In[1]:

import nilearn
import numpy as np
get_ipython().magic('matplotlib inline')


# In[2]:

#first: import nii files into python

# fmri data for training
fmri_filename = '/Users/abby/Desktop/Research/SamplefMRISubject/Inception_20160503_01/11-1-1LocalizerTR1000msSlice44Res2.feat/filtered_func_data.nii'
# fmri run of my dnms task for training
dnms_filename = '/Users/abby/Desktop/Research/SamplefMRISubject/Inception_20160503_01/6-1-1DNMSTR1000msSlice44Res25iso.feat/filtered_func_data.nii'
# mprage
anat_filename = '/Users/abby/Desktop/Research/SamplefMRISubject/Inception_20160503_01/2-1-1t1mpragesagp31mmiso_brain_skull.nii.gz'
# vt mask I made for this subject
mask_filename = '/Users/abby/Desktop/Research/SamplefMRISubject/bilat_vtmask_warped.nii'


# In[3]:

# Visualize the ventral temporal mask
from nilearn import plotting
plotting.plot_roi(mask_filename, anat_filename,
                 cmap='Paired')


# In[ ]:

from nilearn.input_data import NiftiMasker
masker = NiftiMasker(mask_img=mask_filename, standardize=True)

# We give the masker a filename and retrieve a 2D array ready
# for machine learning with scikit-learn
fmri_masked_extra8 = masker.fit_transform(fmri_filename)
dnms_masked_extra8 = masker.fit_transform(dnms_filename)

#good to check the training set fmri_masked and test set dnms_masked have the same number of voxels
#data format = TRs, voxels
print(fmri_masked.shape)
print(dnms_masked.shape)


# In[28]:

#take out first 8 volumes
fmri_masked = fmri_masked_extra8[7 :, :]
dnms_masked = dnms_masked_extra8[7 :, :]
print(fmri_masked_extra8.shape)
print(fmri_masked.shape)


# In[36]:

# Load TR label in formation
import scipy.io as sio
matlabLabels = sio.loadmat('/Users/abby/Desktop/Research/SamplefMRISubject/leftrightfacesceneMatrixSubject39.mat')
labelArray = matlabLabels['StimuliMatrix']
# need to turn the 4*numTrs array into a vector of strings
# rests = 0, leftface = 1, rightface = 2, leftscene = 3, rightscene = 4
maxes = np.max(labelArray, axis=0)
rest_mask = maxes == 0
labelVector_numbers = np.argmax(labelArray, axis=0) + 1
labelVector_numbers[rest_mask] = 0
labelVector_extra4 = labelVector_numbers.astype(str)
print(labelVector_extra4.shape)


# In[42]:

#we push the labels back by 4 TRs to account for hemodynamic lag
hemodynamicLag = np.array(["0", "0", "0", "0"])
labelVector =  np.append(hemodynamicLag, labelVector_extra4)
print(labelVector.shape)

#doing this tanked classifier accuracy, so we'll get rid of it for now
labelVector = labelVector_extra4


# In[43]:

condition_mask = labelVector != "0"   

#take out rests from training set
fmri_masked_norest = fmri_masked[condition_mask]
print(fmri_masked_norest.shape)
condition_norest = labelVector[condition_mask]
print(condition_norest.shape)


# In[11]:

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


# In[46]:

#here we leave 30 TRs out for testing. much better idea!

import sklearn.svm
import sklearn.linear_model
svc_split = sklearn.linear_model.LogisticRegression(multi_class='multinomial', solver='newton-cg')
#svc_split = sklearn.svm.NuSVC(kernel='linear')

svc_split.fit(fmri_masked_norest[:-30,:], condition_norest[:-30])

prediction = svc_split.predict(fmri_masked_norest[-30:])
print((prediction == condition_norest[-30:]).sum() / float(len(condition_norest[-30:])))


# In[45]:

#even better than that, cross validation!!!
from sklearn.cross_validation import KFold

cv = KFold(n=len(fmri_masked_norest), n_folds=5)
target=condition_norest
for train, test in cv:
    svc_split.fit(fmri_masked_norest[train], target[train])
    prediction_split = svc_split.predict(fmri_masked_norest[test])
    print((prediction_split == target[test]).sum() / float(len(target[test])))


# In[40]:

from sklearn.cross_validation import cross_val_score
cv_score = cross_val_score(svc_split, fmri_masked_norest, target)
print(cv_score)


# In[41]:

# i don't know what the cv=cv score flag does....
cv_score = cross_val_score(svc_split, fmri_masked_norest, target, cv=cv)
print(cv_score)


# In[ ]:

#the rest of this is for plotting my coefficient weights
simple_coef = svc_split.coef_
print(simple_coef.shape)

print(simple_coefs.shape)
coef0 = simple_coefs[0, :]
coef1 = simple_coefs[1, :]
coef2 = simple_coefs[2, :]
coef3 = simple_coefs[3, :]

coef_img0 = masker.inverse_transform(coef0)
coef_img1 = masker.inverse_transform(coef1)
coef_img2 = masker.inverse_transform(coef2)
coef_img3 = masker.inverse_transform(coef3)


coef_img0.to_filename('inception0_svc_weights.nii.gz')
coef_img1.to_filename('inception1_svc_weights.nii.gz')
coef_img2.to_filename('inception2_svc_weights.nii.gz')
coef_img3.to_filename('inception3_svc_weights.nii.gz')

#I have 4 classifiers, so that's why I do each of these steps 4 times.


# In[49]:

from nilearn.plotting import plot_stat_map, show

plot_stat_map(coef_img0, anat_filename,
              title="SVM weights", display_mode="yx")

show()


# In[21]:

from nilearn.plotting import plot_stat_map, show

plot_stat_map(coef_img1, anat_filename,
              title="SVM weights", display_mode="yx")

show()


# In[22]:

from nilearn.plotting import plot_stat_map, show

plot_stat_map(coef_img2, anat_filename,
              title="SVM weights", display_mode="yx")

show()


# In[23]:

from nilearn.plotting import plot_stat_map, show

plot_stat_map(coef_img3, anat_filename,
              title="SVM weights", display_mode="yx")

show()


# In[ ]:

# here i test the classifier on new test data. I will use this later to correlate classifier evidence with RTs
# Using predict_log_proba outputs a log probability score for every classifier for every TR
dnms_prediction_split = svc_split.predict_log_proba(dnms_masked)


# In[69]:

# but this is useless without knowing which TRs go with which trials.
dnms_RT = sio.loadmat('/Users/abby/Desktop/Research/SamplefMRISubject/Subject39_AllDNMSRT.mat')
dnms_TR = sio.loadmat('/Users/abby/Desktop/Research/SamplefMRISubject/Subject39_AllDNMSTR.mat')
RT = dnms_RT['DNMSAll']
TR = dnms_TR['TRAll']
print(R)


# In[62]:

runLabels[0, 60]


# In[ ]:



