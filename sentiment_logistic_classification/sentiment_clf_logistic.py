"""
Predicting sentiment from product reviews with a logistic regression classifier
"""
from __future__ import division
import string, json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

def munge(df_input):
    """ Fill in missing values and extract sentiment from reviews"""
    df = df_input.copy()
    assert set(df.columns).issubset(['name', 'review', 'rating'])
    
    # Extract sentiments
    df = df.loc[df['rating'] != 3]        # ignore reviews with rating = 3  
    df['sentiment'] = df['rating'].map(lambda rating: +1 if rating > 3 else -1)
    
    # Remove punctuations from reviews    
    df.fillna({'review': ''}, inplace=True)
    df['review_clean'] = df['review'].map(lambda review: review.translate(None, string.punctuation))
    return df
    
def get_features_matrix(df, vectorizer, learn_vocabulary=True):
    """ Transform reviews into bag-of-words model as specified by vectorizer. 
    Return features_matrix, true_labels as as numpy arrays """ 
    if learn_vocabulary:  # training data
        features_matrix = vectorizer.fit_transform(df['review_clean'])      
    else:       # test data
        features_matrix = vectorizer.transform(df['review_clean'])
 
    output_vector = df['sentiment']    
    return features_matrix, output_vector
    
def fit_logisticRegression(features_matrix, target_vector, vectorizer=None):
    """ Fit a logistic regression model using sklearn implementation.
    If the vectorizer used to construct
    features_matrix is given, return a dataframe containing the weight 
    of each word in the model """    
    model = LogisticRegression()
    model.fit(features_matrix, target_vector)
    
    if vectorizer:
        all_words = vectorizer.vocabulary_.keys()
        weights = model.coef_.flatten()
        weights_table = pd.DataFrame( {'word' : all_words, 
                                       'weight' : weights} )
    else:
        weights_table = None    
    return model, weights_table

def fit_myLogisticRegression(features_matrix, true_labels, \
                             initial_coefficients, l2_penalty=0, \
                             step_size=1e-5, max_iter=1001, verbose=True):
    """ Fit a logistic regression model using the gradient ascent approach.
    Only L2-regularization is supported.
    Note that features_matrix must contain the constant term, i.e feature_(0,i) = 1 
    for all i-th training samples
    """
    coefficients = np.array(initial_coefficients)
    
    # The coefficient update equation is:
    # coef_j += sum_i feature_j * ( ind[y_i = +1] - P(y = +1 | x_i, coef) )
    # where j is the feature index and i is the index of num_samples
    # ind(x) is an indicator function: ind[y_i = +1] = 1 if y_i == +1, 0 if y_i == -1
    for itr in xrange(max_iter):
        # Calculate P(y = +1 | x_i, coef)
        scores = np.dot(features_matrix, coefficients)
        predictions = 1. / (1. + np.exp(-scores))
        
        # Error term is the term inside the brackets on the right-hand side of the coefficient update equation 
        indicators = (true_labels == +1)
        errors = indicators - predictions
        
        # Update each coefficient
        for j in xrange(len(coefficients)):
            derivative = np.dot(features_matrix[:,j], errors).sum()
            if j > 0:
                derivative -= 2 * l2_penalty * coefficients[j]
            coefficients[j] += step_size * derivative
	    
    # Printing the log-likelihood
    if verbose: 
        logexp = -np.log(predictions)
        mask = np.isinf(logexp)              # A check to prevent overflow
        logexp[mask]  = -scores[mask]
        if itr <= 15 or (itr <= 100 and itr % 10 == 0) or (itr <= 1000 and itr % 100 == 0) \
         or (itr <= 10000 and itr % 1000 == 0) or itr % 10000 == 0:
            lp = ((indicators-1) * scores - logexp).sum()
            print 'iteration %*d: log likelihood of observed labels = %.8f' % \
                (int(np.ceil(np.log10(max_iter))), itr, lp)
    return coefficients
	    
def get_classification_accuracy(model, features_matrix, true_labels):
    """ 
    Prediction accuracy of a trained sklearn Logistic Regression model.
    accuracy = num_correct_classification / num_samples
    """
    predictions = model.predict(features_matrix)
    ncorrect = (true_labels - predictions == 0).sum()
    accuracy = float(ncorrect) / len(true_labels)    
    return accuracy
  
def get_classification_accuracy_myLogisticRegression(features_matrix, weights, true_labels):
    """ 
    Prediction accuracy of MyLogisticRegression model.
    accuracy = num_correct_classification / num_samples
    """
    scores = np.dot(features_matrix, weights)
    get_class = np.vectorize(lambda score: +1 if score > 0. else -1)
    predictions = get_class(scores)
    accuracy = (predictions - np.array(true_labels) == 0).sum().astype(float) / len(true_labels)
    return accuracy
    
def getProductsIX_topK_positive_proba(model, features_matrix, K=10):
    scores = model.decision_function(features_matrix)
    predictions_probability = 1. /(1. + np.exp(-scores))
    topK_idx = np.argsort(-predictions_probability)[0:K]
    return topK_idx   
    
def make_coefficient_plot(table, positive_words, negative_words, l2_penalty_list):
    cmap_positive = plt.get_cmap('Reds')
    cmap_negative = plt.get_cmap('Blues')
    
    xx = l2_penalty_list
    plt.plot(xx, [0.]*len(xx), '--', lw=1, color='k')
    
    table_positive_words = table.query('word in @positive_words')
    table_negative_words = table.query('word in @negative_words')
    del table_positive_words['word']
    del table_negative_words['word']
    
    for i in xrange(len(positive_words)):
        color = cmap_positive(0.8*((i+1)/(len(positive_words)*1.2)+0.15))
        plt.plot(xx, table_positive_words[i:i+1].as_matrix().flatten(),
                 '-', label=positive_words[i], linewidth=4.0, color=color)
        
    for i in xrange(len(negative_words)):
        color = cmap_negative(0.8*((i+1)/(len(negative_words)*1.2)+0.15))
        plt.plot(xx, table_negative_words[i:i+1].as_matrix().flatten(),
                 '-', label=negative_words[i], linewidth=4.0, color=color)
        
    plt.legend(loc='best', ncol=3, prop={'size':16}, columnspacing=0.5)
    plt.axis([1, 1e5, -1, 1])
    plt.title('Coefficient path')
    plt.xlabel('L2 penalty ($\lambda$)')
    plt.ylabel('Coefficient value')
    plt.xscale('log')
    plt.rcParams.update({'font.size': 18})
    plt.tight_layout()
    
if __name__ == '__main__':
    # Write the result into a text file
    txtfile = open('test1.txt', 'w')
    
    dtype_dict = {'name' : str, 'review' :str, 'rating': int}
    products = pd.read_csv('../Data/amazon_baby.csv', dtype=dtype_dict)   
    products = munge(products)
    
    # Split the data into 80%, 20% train, test sets
    np.random.seed(1) 
    mask = np.random.rand(len(products)) < 0.8
    train, test = products.iloc[mask], products.iloc[~mask]    
      
    # ------------------------------------------------------------------
    # Case 1: Using all words as features
    # ------------------------------------------------------------------
    txtfile.write('====== MODEL 1: Using all words in the train set as features =====\n')
    
    vectorizer = CountVectorizer(token_pattern=r'\b\w+\b', stop_words='english')
    train_features_matrix, train_labels = get_features_matrix(train, vectorizer, 
                                                               learn_vocabulary=True)
     
    model1, weights1 = fit_logisticRegression(train_features_matrix, train_labels, 
                                     vectorizer=vectorizer)
     
    # Print words that have top 10 largest, positive weights
    top10_poswords = weights1.sort(columns='weight', ascending=False)\
                              .iloc[0:10]
    top10_poswords
         
     # Prediction accuracy on the TRAINING set
    train_accuracy = get_classification_accuracy(model1, 
                                                  train_features_matrix, 
                                                  train_labels)   
    txtfile.write('\nPrediction accuracy of Model1 on the TRAINING set: ' + str(train_accuracy))
     
    # Prediction accuracy on the TEST set
    test_features_matrix, test_labels = get_features_matrix(test, vectorizer, 
                                                               learn_vocabulary=False)
    test_accuracy = get_classification_accuracy(model1, 
                                                test_features_matrix, 
                                                test_labels)
    txtfile.write('\nPrediction accuracy of Model1 on the TEST set: ' + str(test_accuracy))
     
     
    # Top 5 products with highest probability scores of being classified as positive
    top5_ix = getProductsIX_topK_positive_proba(model1, test_features_matrix, K=5)
    txtfile.write('\nTop 5 products predicted by Model1 : ' + str(test.iloc[top5_ix]['name'].tolist()))
     
    # ------------------------------------------------------------------
    # Case 2: Using a simpler classifier with fewer words
    # ------------------------------------------------------------------
    txtfile.write('\n\n====== MODEL 2: A simpler model using 20 words as features =====\n')
    
    significant_words = ['love', 'great', 'easy', 'old', 'little', 'perfect', 
                         'loves', 'well', 'able', 'car', 'broke', 'less', 'even', 
                         'waste', 'disappointed', 'work', 'product', 'money', 
                         'would', 'return']
                         
    vectorizer = CountVectorizer(vocabulary=significant_words)
    
    train_features_matrix, train_labels = get_features_matrix(train, vectorizer, 
                                                              learn_vocabulary=True)
    
    model2, weights2 = fit_logisticRegression(train_features_matrix, train_labels, 
                                    vectorizer=vectorizer)
    
    # Print words that have top 10 largest, positive weights
    top10_poswords = weights2.sort(columns='weight', ascending=False)\
                             .iloc[0:10]
        
    # Prediction accuracy on the TRAINING set
    train_accuracy = get_classification_accuracy(model2, 
                                                 train_features_matrix, 
                                                 train_labels)   
    txtfile.write('\nPrediction accuracy of Model2 on the TRAINING set: ' + str(train_accuracy))
    
    # Prediction accuracy on the TEST set
    test_features_matrix, test_labels = get_features_matrix(test, vectorizer, 
                                                              learn_vocabulary=False)
    test_accuracy = get_classification_accuracy(model2, 
                                                test_features_matrix, 
                                                test_labels)
    txtfile.write('\nPrediction accuracy of Model2 on the TEST set: ' + str(test_accuracy))
    
    # Top 5 products with highest probability of being classified as positive
    top5_ix = getProductsIX_topK_positive_proba(model2, test_features_matrix, K=5)
    txtfile.write('\nTop 5 products predicted by Model2: ' + str(test.iloc[top5_ix]['name'].tolist()))
                                                
    # ------------------------------------------------------------------------
    # Case 3: What about a majority classifier?
    # ------------------------------------------------------------------------                                           
    txtfile.write('\n\n====== A simple majority classifier =====\n')  
    
    num_total = len(train_labels)    
    num_positive = len(filter(lambda x: x == +1, train_labels))
    num_negative = num_total - num_positive
    majclf_accuracy = float(num_positive) / float(num_total) \
                       if num_positive > num_negative \
                       else float(num_negative) / float(num_total)  
                                          
    txtfile.write('\nPrediction accuracy of a majority classifier is: ' + str(majclf_accuracy))

    # ------------------------------------------------------------------------ 
    # Case 4: using myLogisticRegression implementation to fit the model 
    # and explore the effects of L2 penalty
    # ------------------------------------------------------------------------

    txtfile.write('\n\n========= Exploring the effect of l2 penalty==========\n')
    
    # Use a subset of words in the text
    with open('important_words.json', 'r') as f: # Reads the list of most frequent words
        important_words = json.load(f)
    important_words = [str(s) for s in important_words]
    
    vectorizer = CountVectorizer(vocabulary=important_words)

    train_features_matrix, train_labels = get_features_matrix(train, vectorizer, 
                                                               learn_vocabulary=True)
    # The resulting features_matrices are sparse csr matrices.
    # Convert them to array
    train_features_matrix = train_features_matrix.toarray()
    
    # Add a column of ones to both train and test features matrix 
    # Our implementation of Logistic Regression needs a column for the 
    # constant term
    train_features_matrix = np.insert(train_features_matrix, 0, 1, axis=1)
    
    # Split the train set further into 80%, 20% train, validation sets
    np.random.seed(2) 
    mask = np.random.rand(train_features_matrix.shape[0]) < 0.8  
    validation_features_matrix = train_features_matrix[~mask, :]
    validation_labels = train_labels[~mask]
    train_features_matrix = train_features_matrix[mask, :]
    train_labels = train_labels[mask]    
    
    # Store the coefficients for all l2-penalties on a table    
    table = pd.DataFrame({'word': ['(intercept)'] + vectorizer.vocabulary_.keys()})
    
    l2_penalty_list = [0, 4, 10, 1e2, 1e3, 1e5]  
    train_accuracies_l2 = []
    validation_accuracies_l2 = []
    for l2_penalty in l2_penalty_list:  
        initial_weights = np.zeros(train_features_matrix.shape[1])
        coefficients = fit_myLogisticRegression(train_features_matrix, train_labels,
                                                initial_weights, l2_penalty=l2_penalty, 
                                                step_size=1e-6, max_iter=501,
                                                verbose=False)
        column_name = 'coefficients [L2=' + str(l2_penalty) + ']'
        table[column_name] = coefficients
        
        train_accuracy = get_classification_accuracy_myLogisticRegression(train_features_matrix, coefficients, train_labels)
        train_accuracies_l2.append(train_accuracy)
        
        validation_accuracy = get_classification_accuracy_myLogisticRegression(validation_features_matrix, coefficients, validation_labels)
        validation_accuracies_l2.append(validation_accuracy)
    
    table_sorted = table[['word', 'coefficients [L2=0]']].query('word != "(intercept)"') \
                                                         .sort('coefficients [L2=0]', ascending=False)
    positive_words = table_sorted.iloc[0:5]['word'].values.tolist()
    negative_words = table_sorted.iloc[-6:-1]['word'].values.tolist()
    
    # Plot the coefficients of top 5 positive and negative words
    # as a function of l2 penalty
    make_coefficient_plot(table, positive_words, negative_words, l2_penalty_list)                                              
        
    plt.savefig('l2_coefficients.png', bbox_inches='tight', dpi=150)
    txtfile.write("\nPrinting coefficient path to l2_coefficients.png\n")    
    plt.cla()
    
    # Report the prediction accuracy of the train and validation sets
    # as a function of L2 penalty
    txtfile.write("Prediction accuracies of the train and validation set with varying L2 penalty:\n")
    for l2_penalty, acc_tr, acc_val in zip(l2_penalty_list, train_accuracies_l2, validation_accuracies_l2):
        txtfile.write("L2 penalty = %g \n" %l2_penalty)
        txtfile.write("train accuracy = %s, validation_accuracy = %s \n" % (acc_tr, acc_val))
        txtfile.write("--------------------------------------------------------------------------------\n")
    
    # Close textfile
    txtfile.close()