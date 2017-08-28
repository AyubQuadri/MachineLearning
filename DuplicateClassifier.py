import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.ensemble.forest import RandomForestClassifier
from sklearn.metrics import f1_score,accuracy_score,recall_score,log_loss,precision_score
from sklearn.ensemble.weight_boosting import AdaBoostClassifier
import xgboost as xgb
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import VotingClassifier
# from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA
from sknn.mlp import Classifier, Layer
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression

def readData(path):
    data = pd.DataFrame.from_csv(path, header=True, index_col=False)
    return data

def splitData(data,testSize):
  np.random.seed(10)
  trainData,testData = train_test_split(data, test_size = testSize)
  return trainData,testData

def executePCA(data):
    ## code to find out the n through manual inspection
    # pca = PCA(n_components=50)
    # pca.fit(data.values)
    # var= pca.explained_variance_ratio_
    #
    # print var
    # #Cumulative Variance explains
    # var1=np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)
    #
    # print var1
    #
    # plt.plot(var1)
    #
    # plt.show()

    pca = PCA(n_components=10)
    pca.fit(data.values)
    pcaData=pca.fit_transform(data.values)
    return pcaData

def RandomForestClassification(trainFeatures,trainTarget,testFeatures,testTarget):
    #
    rf = RandomForestClassifier(n_estimators=1000,oob_score=True,max_features='sqrt',n_jobs=4,min_samples_leaf=1)
    rf.fit(trainFeatures,trainTarget)

    predictions =rf.predict(testFeatures)
    print 'f1 - random forest', f1_score(testTarget, predictions)
    # print 'time - scikit', (endTime- startTime)
    return rf


def AdaBoostClassification(trainFeatures,trainTarget,testFeatures,testTarget):

    ab = AdaBoostClassifier(n_estimators=100,learning_rate=0.5)
    ab.fit(trainFeatures,trainTarget)

    predictions =ab.predict(testFeatures)

    print 'f1 - adaboost', f1_score(testTarget, predictions)

    return ab

def xgBoostClassification(trainFeatures,trainTarget,testFeatures,testTarget,maxDepth,learningrate,estimators,minChildweight):

    gbm = xgb.XGBClassifier( learning_rate =0.1, n_estimators=800, max_depth=2,min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27)

    gbm.fit(trainFeatures, trainTarget)

    joblib.dump(gbm,"data/gbm.pkl")

    # gbm = xgb.XGBClassifier(max_depth=maxDepth, n_estimators=estimators, learning_rate=learningrate,min_child_weight=minChildweight).fit(trainFeatures, trainTarget)

    predictions =gbm.predict(testFeatures)
    predictionProbabilities = gbm.predict_proba(testFeatures)

    predictionSet = pd.concat([pd.DataFrame(predictionProbabilities) ,pd.DataFrame(predictions)],axis=1)
    # predictionSet.to_csv("data/submission_" + "probabilities" +".csv",index=False)

    # print 'f1 - xgboost', accuracy_score(testTarget, predictions)
    # print 'recall - xgboost', recall_score(testTarget, predictions)

    print 'accuracy - xgboost', accuracy_score(testTarget, predictions)
    print 'precision - xgboost', precision_score(testTarget, predictions)
    print 'recall - xgboost', recall_score(testTarget, predictions)
    print 'f1 - xgboost', f1_score(testTarget, predictions)

    return gbm

def xgBoostClassificationProbabilities(trainFeatures,trainTarget,testFeatures,testTarget,maxDepth,learningrate,estimators,minChildweight):

    gbm = xgb.XGBClassifier( learning_rate =0.1, n_estimators=800, max_depth=2,min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27)

    gbm.fit(trainFeatures, trainTarget,eval_metric='logloss')

    joblib.dump(gbm, "data/gbModel.pkl")

    # gbm = xgb.XGBClassifier(max_depth=maxDepth, n_estimators=estimators, learning_rate=learningrate,min_child_weight=minChildweight).fit(trainFeatures, trainTarget)

    predictions =gbm.predict(testFeatures)
    predictionProbabilities = gbm.predict_proba(testFeatures)

    predictionSet = pd.concat([pd.DataFrame(predictionProbabilities) ,pd.DataFrame(predictions)],axis=1)
    predictionSet.to_csv("data/submission_" + "probabilities" +".csv",index=False)

    print 'accuracy - xgboost', accuracy_score(testTarget, predictions)
    print 'precision - xgboost', precision_score(testTarget, predictions)
    print 'recall - xgboost', recall_score(testTarget, predictions)
    print 'f1 - xgboost', f1_score(testTarget, predictions)

    return gbm


def SVMClassification(trainFeatures,trainTarget,testFeatures,testTarget):

    svm = SVC().fit(trainFeatures, trainTarget)
    predictions =svm.predict(testFeatures)
    print 'f1 - svm', accuracy_score(testTarget, predictions)
    # print 'time - scikit', (endTime- startTime)
    return svm


def predictData(model,modelType):

    submissionData = readData("data/TestDataTwoClass.csv")
    ids = pd.DataFrame(submissionData['Id'])

    features =submissionData.drop(['Id'],axis=1)
    # pca_features=executePCA(features)
    predictions = pd.DataFrame(model.predict(features))

    predictions.columns = ['class']

    submissionSet = pd.concat([ids,predictions],axis=1)
    submissionSet.to_csv("data/submission_" + modelType +".csv",index=False,columns=['Id','class'])

def predictProbabilities(model,modelType):

    def getClass (row):
       if row['probability1'] > 0.15 :
          return 1
       else:
           return 0

    submissionData = readData("data/TestDataTwoClass.csv")
    ids = pd.DataFrame(submissionData['Id'])

    features =submissionData.drop(['Id'],axis=1)
    # pca_features=executePCA(features)
    # predictions = pd.DataFrame(model.predict(features))
    # predictions.columns = ['class']
    predictionProbs = pd.DataFrame(model.predict_proba(features))
    predictionProbs.columns=['probability0','probability1']
    predictionProbs['class'] = predictionProbs.apply (lambda row: getClass (row),axis=1)
    submissionSet = pd.concat([ids,predictionProbs['class']],axis=1)
    submissionSet.to_csv("data/submission_" + modelType +".csv",index=False,columns=['Id','class'])



def NaiveBayesClassification(trainFeatures,trainTarget,testFeatures,testTarget):

    nb = GaussianNB()
    nb.fit(trainFeatures,trainTarget)
    predictions =nb.predict(testFeatures)

    print 'f1 - Naive Bayes', f1_score(testTarget, predictions)
    return nb

def VotingClassification(trainFeatures,trainTarget,testFeatures,testTarget):

    nb = GaussianNB()
    svm = SVC(kernel='sigmoid',degree=10,class_weight='balanced',max_iter=100)
    gbm = xgb.XGBClassifier(max_depth=9, n_estimators=430, learning_rate=0.1,min_child_weight=1)
    ab = AdaBoostClassifier(n_estimators=400,learning_rate=0.5)
    rf = RandomForestClassifier(n_estimators=400)

    em = VotingClassifier(estimators=[('nb',nb),('svm',svm),('gbm',gbm),('ab',ab),('rf',rf)], voting='hard').fit(trainFeatures,trainTarget)

    predictions =em.predict(testFeatures)

    print 'f1 - Ensemble', f1_score(testTarget, predictions)

    return em


def neuralnetForWeightedClassification(models, trainFeatures,trainTarget,testFeatures,testTarget):

    predictionsTable =pd.DataFrame()

    for model in models:
        prediction = pd.DataFrame(model.predict(trainFeatures))
        predictionsTable = pd.concat([predictionsTable,prediction],axis=1)

    X_train =predictionsTable.values
    y_train= trainTarget
    nn = Classifier(layers=[Layer("Sigmoid", units=100),Layer("Softmax")],learning_rate=0.02,n_iter=5)
    nn.fit(X_train, y_train)

    predictionsTable =pd.DataFrame()

    for model in models:
        prediction = pd.DataFrame(model.predict(testFeatures))
        predictionsTable = pd.concat([predictionsTable,prediction],axis=1)

    X_valid= predictionsTable
    y_valid = nn.predict(X_valid)

    print 'f1 - neuralnetEnsemble', f1_score(testTarget, y_valid)
    return nn

# def smoteData(data):
#     sm = SMOTE(kind='svm')
#     trainTarget = data['class']
#     trainFeatures = data.drop(['class'],axis=1)
#     colNames = list(trainFeatures.columns.values)
#     X_resampled, y_resampled = sm.fit_sample(trainFeatures, trainTarget)
#     smotedFeatures =pd.DataFrame(X_resampled)
#     smotedFeatures.columns = colNames
#     smotedLabels = pd.DataFrame(y_resampled)
#     smotedLabels.columns=['class']
#     smotedDataset= pd.concat([smotedFeatures,smotedLabels ],axis=1)
#
#     return smotedDataset



milkData = pd.read_csv("/home/synerzip/Sasidhar/Mtech/Fourth Sem/Advanced ML/Project/DuplicateQuestionDetection/data/quora_features_all.csv",sep=",")


# print len(milkData.loc[milkData["class"]==1])
# print len(milkData.loc[milkData["class"]==0])
#
# # smotedData = smoteData(milkData)
#
smotedData = milkData
#
# print len(smotedData.loc[smotedData["class"]==1])
# print len(smotedData.loc[smotedData["class"]==0])
#
#
# milkData = milkData.drop(['Id'],axis=1)
# smotedData = smotedData.drop(['Id'],axis=1)
#
#
# trainData,testData = splitData(smotedData,0.2)
milkData = smotedData
milkData = milkData.drop(['question1'],axis=1).drop(['question2'],axis=1)

train,test = splitData(milkData,0.2)
#
trainLabels = train['is_duplicate']
trainData = train.drop(['is_duplicate'],axis=1)
#
testLabels = test['is_duplicate']
testData = test.drop(['is_duplicate'],axis=1)
#
# # pca_trainData = pd.concat([pd.DataFrame(executePCA(trainData)),pd.DataFrame(trainLabels)],axis=1)
# # pca_testData = pd.concat([pd.DataFrame(executePCA(testData)),pd.DataFrame(testLabels)],axis=1)
#
# # pca_trainData = executePCA(trainData)
# # pca_testData = executePCA(testData)
#
# rfModel =RandomForestClassification(trainData,trainLabels,testData,testLabels)
#
#
# # train model on the full training set with tuned hyperparameters
#
def predictDataforNN(nnModel,models,modelType):
    # testTarget =testData['class']
    submissionData = readData("data/TestDataTwoClass.csv")
    ids = pd.DataFrame(submissionData['Id'])

    features =submissionData.drop(['Id'],axis=1)

    predictionsTable =pd.DataFrame()

    for model in models:
        prediction = pd.DataFrame(model.predict(features))
        predictionsTable = pd.concat([predictionsTable,prediction],axis=1)

    X_valid= predictionsTable
    predictions = pd.DataFrame(nnModel.predict(X_valid))
    predictions.columns = ['class']
    submissionSet = pd.concat([ids,predictions],axis=1)
    submissionSet.to_csv("data/submission_" + modelType +".csv",index=False,columns=['Id','class'])

def predictDataforLogisticRegression(nnModel,models,modelType):
    # testTarget =testData['class']
    submissionData = readData("data/TestDataTwoClass.csv")
    ids = pd.DataFrame(submissionData['Id'])

    features =submissionData.drop(['Id'],axis=1)

    predictionsTable =pd.DataFrame()

    for model in models:
        prediction = pd.DataFrame(model.predict(features))
        predictionsTable = pd.concat([predictionsTable,prediction],axis=1)

    X_valid= predictionsTable
    predictions = pd.DataFrame(nnModel.predict(X_valid))
    predictions.columns = ['class']
    submissionSet = pd.concat([ids,predictions],axis=1)
    submissionSet.to_csv("data/submission_" + modelType +".csv",index=False,columns=['Id','class'])

    # print 'f1 - neuralnetEnsemble', f1_score(testTarget, y_valid)


milkLables = milkData['is_duplicate']
milkFeatures = milkData.drop(['is_duplicate'],axis=1)
#
xgbModel = xgBoostClassification(trainData,trainLabels,testData,testLabels,8,0.1,430,1)
# svmModel = SVMClassification(milkFeatures,milkLables,testData,testLabels)
# nbModel = NaiveBayesClassification(trainData,trainLabels,testData,testLabels)
# rfModel = RandomForestClassification(milkFeatures,milkLables,testData,testLabels)
#
# abModel =AdaBoostClassification(trainData,trainLabels,testData,testLabels)
# # #
# #
# nnEnsembler= neuralnetForWeightedClassification([rfModel,xgbModel,svmModel,abModel],trainData,trainLabels,testData,testLabels)
#
# # predictProbabilities(rfModel,"rf_prob")
#
# #
# predictDataforNN(nnEnsembler,[rfModel,xgbModel,svmModel,abModel],"nn_ensembler_no_nb_fulldata")
# # predictDatausingLogisticRegression([rfModel,xgbModel,svmModel],"nn_ensembler_no_nb")
#
# # neuralnetForWeightedClassification([nbModel,svmModel],trainData,trainLabels,testData,testLabels)
# # neuralnetForWeightedClassification([rfModel,nbModel,xgbModel,abModel,svmModel],trainData,trainLabels,testData,testLabels)
#
#
#
# # emModel = VotingClassification(milkFeatures,milkLables,testData,testLabels)
#
# # predictData(xgbModel,"xgb_smoting_probabilities")
# # predictData(rfModel,"rf_1500_smoted")
# # predictData(nbModel,"nb")
# #  predictData(svmModel,"svm")
# # predictData(emModel,"em_smoting")
#
