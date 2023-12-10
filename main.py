import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier, export_text, export_graphviz
from statsmodels.graphics.gofplots import qqplot
import scipy.stats as stats
from scipy.stats import pearsonr, spearmanr, norm, chi2_contingency, kurtosis, skew
from sklearn.covariance import EllipticEnvelope
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score
from scipy.stats import anderson
#KNN Class import from Scikit Learn
from sklearn.neighbors import KNeighborsClassifier
#Random Forest class import from Scikit Learn
from sklearn.ensemble import RandomForestClassifier

data = pd.read_csv('telecom_churn.csv')
print(data.columns)
print(data.describe())
print(data.dtypes)

# Converts data to numbers - errors='coerse' is used for any data that can't be converted - NaN
data['Account length'] = pd.to_numeric(data['Account length'], errors='coerce')
data['Area code'] = pd.to_numeric(data['Area code'], errors='coerce')
data['Number vmail messages'] = pd.to_numeric(data['Number vmail messages'], errors='coerce')
data['Total day calls'] = pd.to_numeric(data['Total day calls'], errors='coerce')
data['Total eve calls'] = pd.to_numeric(data['Total eve calls'], errors='coerce')
data['Total night calls'] = pd.to_numeric(data['Total night calls'], errors='coerce')
data['Total intl calls'] = pd.to_numeric(data['Total intl calls'], errors='coerce')
data['Customer service calls'] = pd.to_numeric(data['Customer service calls'], errors='coerce')

print(data.info())
print(data.isnull().sum())

#The percentage of each state in the dataset
state_counts = data['State'].value_counts(normalize=True) * 100
plt.figure(figsize=(10, 6))
sns.barplot(x=state_counts.index, y=state_counts.values)
plt.xlabel('Πολιτεία')
plt.ylabel('Ποσοστό (%)')
plt.title('Πολιτεία')
plt.xticks(rotation=90)
plt.show()

time.sleep(5)

#The percentage of the people having vs not having international plan in the dataset
international_plan_counts = (data['International plan'].value_counts(normalize=True) * 100).sort_index()
plt.figure(figsize=(10, 6))
plt.bar(international_plan_counts.index, international_plan_counts.values, color='blue')
plt.xlabel('International Plan')
plt.ylabel('Ποσοστό (%)')
plt.title('Ποσοστό International Plan')
plt.xticks(rotation=90)
plt.show()

time.sleep(5)

#The percentage of people having vs not having a voice mail plan in the dataset
voice_mail_plan_counts = (data['Voice mail plan'].value_counts(normalize=True) * 100).sort_index()
plt.figure(figsize=(10, 6))
plt.bar(voice_mail_plan_counts.index, voice_mail_plan_counts.values, color='green')
plt.xlabel('Voice Mail Plan')
plt.ylabel('Ποσοστό (%)')
plt.title('Ποσοστό Voice Mail Plan')
plt.xticks(rotation=90)
plt.show()

time.sleep(5)

#The percentage of churn values in the dataset
churn_counts = (data['Churn'].value_counts(normalize=True) * 100).sort_index()
plt.figure(figsize=(6, 6))
plt.pie(churn_counts.values, labels=churn_counts.index, autopct='%1.1f%%', startangle=140)
plt.title('Ποσοστό Churn')
plt.show()

time.sleep(5)

#The frequency of each Account Length
plt.figure(figsize=(10, 6))
plt.hist(data['Account length'], bins=30, color='gray', alpha=0.7, edgecolor='black')
plt.axvline(np.mean(data['Account length']), color='red', linestyle='dashed', linewidth=2, label='Μέση')
plt.axvline(np.median(data['Account length']), color='blue', linestyle='dashed', linewidth=2, label='Διάμεσο')
plt.xlabel('Μήκος Λογαριασμού')
plt.ylabel('Συχνότητα')
plt.title('Κατανομή Μήκους Λογαριασμού')
plt.legend()
plt.show()
plt.figure(figsize=(8, 6))
stats.probplot(data['Account length'], dist='norm', plot=plt)
plt.title('QQ-plot για Μήκος Λογαριασμού')
plt.show()

time.sleep(5)

#The frequency of each Area code
plt.figure(figsize=(10, 6))
plt.hist(data['Area code'], bins=30, color='gray', alpha=0.7, edgecolor='black')
plt.axvline(np.mean(data['Area code']), color='red', linestyle='dashed', linewidth=2, label='Μέση')
plt.axvline(np.median(data['Area code']), color='blue', linestyle='dashed', linewidth=2, label='Διάμεσο')
plt.xlabel('Κωδικός Περιοχής')
plt.ylabel('Συχνότητα')
plt.title('Κατανομή Κωδικού Περιοχής')
plt.legend()
plt.show()
plt.figure(figsize=(8, 6))
stats.probplot(data['Area code'], dist='norm', plot=plt)
plt.title('QQ-plot για Κωδικό Περιοχής')
plt.show()

time.sleep(5)

#The frequency of total day minutes
plt.figure(figsize=(10, 6))
plt.hist(data['Total day minutes'], bins=30, color='gray', alpha=0.7, edgecolor='black')
plt.axvline(np.mean(data['Total day minutes']), color='red', linestyle='dashed', linewidth=2, label='Μέση')
plt.axvline(np.median(data['Total day minutes']), color='blue', linestyle='dashed', linewidth=2, label='Διάμεσο')
plt.xlabel('Συνολικά Λεπτά Ημέρας')
plt.ylabel('Συχνότητα')
plt.title('Κατανομή Συνολικών Λεπτών Ημέρας')
plt.legend()
plt.show()
plt.figure(figsize=(8, 6))
stats.probplot(data['Total day minutes'], dist='norm', plot=plt)
plt.title('QQ-plot για Συνολικά Λεπτά Ημέρας')
plt.show()

time.sleep(5)

#The frequency of total day calls
plt.figure(figsize=(10, 6))
plt.hist(data['Total day calls'], bins=30, color='gray', alpha=0.7, edgecolor='black')
plt.axvline(np.mean(data['Total day calls']), color='red', linestyle='dashed', linewidth=2, label='Μέση')
plt.axvline(np.median(data['Total day calls']), color='blue', linestyle='dashed', linewidth=2, label='Διάμεσο')
plt.xlabel('Συνολικά Τηλέφωνα Ημέρας')
plt.ylabel('Συχνότητα')
plt.title('Κατανομή Συνολικών Τηλεφώνων Ημέρας')
plt.legend()
plt.show()
plt.figure(figsize=(8, 6))
stats.probplot(data['Total day calls'], dist='norm', plot=plt)
plt.title('QQ-plot για Συνολικά Τηλέφωνα Ημέρας')
plt.show()

time.sleep(5)

#The frequency of total day charge
plt.figure(figsize=(10, 6))
plt.hist(data['Total day charge'], bins=30, color='gray', alpha=0.7, edgecolor='black')
plt.axvline(np.mean(data['Total day charge']), color='red', linestyle='dashed', linewidth=2, label='Μέση')
plt.axvline(np.median(data['Total day charge']), color='blue', linestyle='dashed', linewidth=2, label='Διάμεσο')
plt.xlabel('Συνολική Χρέωση Ημέρας')
plt.ylabel('Συχνότητα')
plt.title('Κατανομή Συνολικής Χρέωσης Ημέρας')
plt.legend()
plt.show()
plt.figure(figsize=(8, 6))
stats.probplot(data['Total day charge'], dist='norm', plot=plt)
plt.title('QQ-plot για Συνολική Χρέωση Ημέρας')
plt.show()

time.sleep(5)

#The frequency of total eve calls
plt.figure(figsize=(10, 6))
plt.hist(data['Total eve calls'], bins=30, color='gray', alpha=0.7, edgecolor='black')
plt.axvline(np.mean(data['Total eve calls']), color='red', linestyle='dashed', linewidth=2, label='Μέση')
plt.axvline(np.median(data['Total eve calls']), color='blue', linestyle='dashed', linewidth=2, label='Διάμεσο')
plt.xlabel('Συνολικά Τηλέφωνα Απογεύματος')
plt.ylabel('Συχνότητα')
plt.title('Κατανομή Συνολικών Τηλεφώνων Απογεύματος')
plt.legend()
plt.show()
plt.figure(figsize=(8, 6))
stats.probplot(data['Total eve calls'], dist='norm', plot=plt)
plt.title('QQ-plot για Συνολικά Τηλέφωνα Απογεύματος')
plt.show()

time.sleep(5)

#The frequency of total eve minutes
plt.figure(figsize=(10, 6))
plt.hist(data['Total eve minutes'], bins=30, color='gray', alpha=0.7, edgecolor='black')
plt.axvline(np.mean(data['Total eve minutes']), color='red', linestyle='dashed', linewidth=2, label='Μέση')
plt.axvline(np.median(data['Total eve minutes']), color='blue', linestyle='dashed', linewidth=2, label='Διάμεσο')
plt.xlabel('Συνολικά Λεπτά Απογεύματος')
plt.ylabel('Συχνότητα')
plt.title('Κατανομή Συνολικών Λεπτών Απογεύματος')
plt.legend()
plt.show()
plt.figure(figsize=(8, 6))
stats.probplot(data['Total eve minutes'], dist='norm', plot=plt)
plt.title('QQ-plot για Συνολικά Λεπτά Απογεύματος')
plt.show()

time.sleep(5)

#The frequency of total eve charge
plt.figure(figsize=(10, 6))
plt.hist(data['Total eve charge'], bins=30, color='gray', alpha=0.7, edgecolor='black')
plt.axvline(np.mean(data['Total eve charge']), color='red', linestyle='dashed', linewidth=2, label='Μέση')
plt.axvline(np.median(data['Total eve charge']), color='blue', linestyle='dashed', linewidth=2, label='Διάμεσο')
plt.xlabel('Συνολική Χρέωση Απογεύματος')
plt.ylabel('Συχνότητα')
plt.title('Κατανομή Συνολικής Χρέωσης Απογεύματος')
plt.legend()
plt.show()
plt.figure(figsize=(8, 6))
stats.probplot(data['Total eve charge'], dist='norm', plot=plt)
plt.title('QQ-plot για Συνολική Χρέωση Απογεύματος')
plt.show()

time.sleep(5)

#The frequency of total night minutes
plt.figure(figsize=(10, 6))
plt.hist(data['Total night minutes'], bins=30, color='gray', alpha=0.7, edgecolor='black')
plt.axvline(np.mean(data['Total night minutes']), color='red', linestyle='dashed', linewidth=2, label='Μέση')
plt.axvline(np.median(data['Total night minutes']), color='blue', linestyle='dashed', linewidth=2, label='Διάμεσο')
plt.xlabel('Συνολικά Λεπτά το Βράδυ')
plt.ylabel('Συχνότητα')
plt.title('Κατανομή Συνολικών Λεπτών το Βράδυ')
plt.legend()
plt.show()
plt.figure(figsize=(8, 6))
stats.probplot(data['Total night minutes'], dist='norm', plot=plt)
plt.title('QQ-plot για Συνολικά Λεπτά το Βράδυ')
plt.show()

time.sleep(5)

#The frequency of total night calls
plt.figure(figsize=(10, 6))
plt.hist(data['Total night calls'], bins=30, color='gray', alpha=0.7, edgecolor='black')
plt.axvline(np.mean(data['Total night calls']), color='red', linestyle='dashed', linewidth=2, label='Μέση')
plt.axvline(np.median(data['Total night calls']), color='blue', linestyle='dashed', linewidth=2, label='Διάμεσο')
plt.xlabel('Συνολικά Τηλέφωνα το Βράδυ')
plt.ylabel('Συχνότητα')
plt.title('Κατανομή Συνολικών Τηλεφώνων το Βράδυ')
plt.legend()
plt.show()
plt.figure(figsize=(8, 6))
stats.probplot(data['Total night calls'], dist='norm', plot=plt)
plt.title('QQ-plot για Συνολικά Τηλέφωνα το Βράδυ')
plt.show()

time.sleep(5)

#The frequency of total night charge
plt.figure(figsize=(10, 6))
plt.hist(data['Total night charge'], bins=30, color='gray', alpha=0.7, edgecolor='black')
plt.axvline(np.mean(data['Total night charge']), color='red', linestyle='dashed', linewidth=2, label='Μέση')
plt.axvline(np.median(data['Total night charge']), color='blue', linestyle='dashed', linewidth=2, label='Διάμεσο')
plt.xlabel('Συνολική Χρέωση το Βράδυ')
plt.ylabel('Συχνότητα')
plt.title('Κατανομή Συνολικής Χρέωσης το Βράδυ')
plt.legend()
plt.show()
plt.figure(figsize=(8, 6))
stats.probplot(data['Total night charge'], dist='norm', plot=plt)
plt.title('QQ-plot για Συνολική Χρέωση το Βράδυ')
plt.show()

time.sleep(5)

#The frequency of total international calls
plt.figure(figsize=(10, 6))
plt.hist(data['Total intl calls'], bins=30, color='gray', alpha=0.7, edgecolor='black')
plt.axvline(np.mean(data['Total intl calls']), color='red', linestyle='dashed', linewidth=2, label='Μέση')
plt.axvline(np.median(data['Total intl calls']), color='blue', linestyle='dashed', linewidth=2, label='Διάμεσο')
plt.xlabel('Συνολικός Αριθμός Διεθνών κλήσεων')
plt.ylabel('Συχνότητα')
plt.title('Συνολικός Αριθμός Διεθνών κλήσεων')
plt.legend()
plt.show()
plt.figure(figsize=(8, 6))
stats.probplot(data['Total intl calls'], dist='norm', plot=plt)
plt.title('QQ-plot για Συνολικό Αριθμό Διεθνών κλήσεων')
plt.show()

time.sleep(5)

#The frequency of total international minutes
plt.figure(figsize=(10, 6))
plt.hist(data['Total intl minutes'], bins=30, color='gray', alpha=0.7, edgecolor='black')
plt.axvline(np.mean(data['Total intl minutes']), color='red', linestyle='dashed', linewidth=2, label='Μέση')
plt.axvline(np.median(data['Total intl minutes']), color='blue', linestyle='dashed', linewidth=2, label='Διάμεσο')
plt.xlabel('Συνολικά Λεπτά Διεθνών Κλήσεων')
plt.ylabel('Συχνότητα')
plt.title('Συνολικά Λεπτά Διεθνών κλήσεων')
plt.legend()
plt.show()
plt.figure(figsize=(8, 6))
stats.probplot(data['Total intl minutes'], dist='norm', plot=plt)
plt.title('QQ-plot για Συνολικά Λεπτά Διεθνών κλήσεων')
plt.show()

time.sleep(5)

#The frequency of total international charge
plt.figure(figsize=(10, 6))
plt.hist(data['Total intl charge'], bins=30, color='gray', alpha=0.7, edgecolor='black')
plt.axvline(np.mean(data['Total intl charge']), color='red', linestyle='dashed', linewidth=2, label='Μέση')
plt.axvline(np.median(data['Total intl charge']), color='blue', linestyle='dashed', linewidth=2, label='Διάμεσο')
plt.xlabel('Συνολική Χρέωση Διεθνών Κλήσεων')
plt.ylabel('Συχνότητα')
plt.title('Συνολική Χρέωση Διεθνών κλήσεων')
plt.legend()
plt.show()
plt.figure(figsize=(8, 6))
stats.probplot(data['Total intl charge'], dist='norm', plot=plt)
plt.title('QQ-plot για Συνολική Χρέωση Διεθνών κλήσεων')
plt.show()

time.sleep(5)

#The frequency of customer service calls
plt.figure(figsize=(10, 6))
plt.hist(data['Customer service calls'], bins=30, color='gray', alpha=0.7, edgecolor='black')
plt.axvline(np.mean(data['Customer service calls']), color='red', linestyle='dashed', linewidth=2, label='Μέση')
plt.axvline(np.median(data['Customer service calls']), color='blue', linestyle='dashed', linewidth=2, label='Διάμεσο')
plt.xlabel('Συνολικός Αριθμός Τηλεφώνων προς την Εξυπηρέτηση Πελατών')
plt.ylabel('Συχνότητα')
plt.title('Συνολικός Αριθμός Τηλεφώνων προς την Εξυπηρέτηση Πελατών')
plt.legend()
plt.show()
plt.figure(figsize=(8, 6))
stats.probplot(data['Customer service calls'], dist='norm', plot=plt)
plt.title('QQ-plot για Συνολικό Αριθμό Τηλεφώνων προς την Εξυπηρέτηση Πελατών')
plt.show()

time.sleep(5)


# Correlation
numeric_columns = data.select_dtypes(include=np.number).columns
summary_stats = data[numeric_columns].describe().transpose()
print(round(summary_stats, 2))

rounded_data = np.round(data[numeric_columns], 2)
print("Rounded data:")
print(rounded_data.describe())

# Υπολογισμός πίνακα συσχέτισης Pearson
correlation_pearson = data[numeric_columns].corr(method='pearson')
print("Correlation Pearson:")
print(correlation_pearson)

# Σχεδίαση πίνακα συσχέτισης
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_pearson, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Pearson Correlation Heatmap")
plt.show()

# Υπολογισμός πίνακα συσχέτισης Spearman
correlation_spearman = data[numeric_columns].corr(method='spearman')
print(correlation_spearman)

# Σχεδίαση πίνακα συσχέτισης Spearman
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_spearman, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Spearman Correlation Heatmap")
plt.show()

# Υπολογισμός συσχέτισης και εκτέλεση του τεστ
correlation_minutes_charge = pearsonr(data['Total day minutes'], data['Total day charge'])
print(f"Correlation between Total day minutes and Total day charge: {correlation_minutes_charge[0]:.4f}")

# Τεστ συσχέτισης με χρήση cor.test
cor_test_minutes_charge = spearmanr(data['Total day minutes'], data['Total day charge'])
print(cor_test_minutes_charge)

# Εύρεση εκτός των κύριων ακτίνων με Mahalanobis
features = ['Total day minutes', 'Total day calls', 'Total day charge']
envelope = EllipticEnvelope()
envelope.fit(data[features])
outliers = envelope.predict(data[features])

# Ποσοστό εκτός των κύριων ακτίνων
outlier_percentage = (sum(outliers == -1) / len(data)) * 100
print(f"Percentage of outliers: {outlier_percentage:.1f}%")


# Εμφάνιση boxplot
plt.figure(figsize=(10, 6))
sns.boxplot(data=data[numeric_columns])
plt.title("Boxplot of Numeric Variables")
plt.xticks(rotation=45)
plt.show()

# Εύρεση ακραίων τιμών
for col in numeric_columns:
# Πάρτε τις γραμμές του boxplot
     lines = sns.boxplot(data=data[col], showfliers=False).get_lines()

# Πάρτε τα δεδομένα των ακραίων τιμών
     outliers = lines[0].get_ydata()

     if len(outliers) != 0:
         print('-------------------------------------------------------')
         print(f'Outliers for variable {col}')
         print(f'{len(outliers)} outliers')
         print(f'{round(100 * len(outliers) / len(data[col]), 1)}% outliers')
         print(outliers)


# --- Εκτέλεση των Chi-Square Tests και εμφάνιση των contingency tables

def chi_square_test(cross_tab):
     chi2, p, _, _ = chi2_contingency(cross_tab)
     print(f"Chi-Square Value: {chi2:.4f}")
     print(f"P-value: {p:.4f}")
     print("")


# Churn vs State
cross_tab_state = pd.crosstab(data['Churn'], data['State'])
chi_square_test(cross_tab_state)

# Churn vs International Plan
cross_tab_international_plan = pd.crosstab(data['Churn'], data['International plan'])
chi_square_test(cross_tab_international_plan)

# Churn vs Voice Mail Plan
cross_tab_voice_mail_plan = pd.crosstab(data['Churn'], data['Voice mail plan'])
chi_square_test(cross_tab_voice_mail_plan)

# -----------------------------------------------------------------------------

#--- Skewness & kurtosis

skewness = data[numeric_columns].apply(skew)
kurt = data[numeric_columns].apply(kurtosis)

print("Skewness:")
print(skewness)

print("\nKurtosis:")
print(kurt)


# --- Testing for normality
numeric_columns = data.select_dtypes(include=np.number)
y = numeric_columns

for col in y.columns:
    ks_stat, ks_p_value = stats.kstest(y[col], 'norm')
    print(f"Kolmogorov-Smirnov test for {col}: KS Statistic = {ks_stat}, p-value = {ks_p_value}")

anderson_test_results = y.apply(lambda x: anderson(x).statistic)
print("Anderson-Darling test results:")
print(anderson_test_results)

shapiro_test_results = y.apply(stats.shapiro)
print("Shapiro-Wilk test results:")
print(shapiro_test_results)

t_test_results = [stats.ttest_1samp(y[col], 0) for col in y.columns]
print("One-sample t-test results:")
print(t_test_results)



# Logistic Regression

# Διαχωρισμός των ανεξάρτητων (X) και εξαρτημένης (y) μεταβλητής
data_encoded = pd.get_dummies(data, columns=['International plan', 'Voice mail plan', 'State'])

# Διαχωρισμός των ανεξάρτητων (X) και εξαρτημένης (y) μεταβλητής
X = data_encoded.drop('Churn', axis=1)
y = data_encoded['Churn']

# Ορισμός του μοντέλου Logistic Regression
logreg = LogisticRegression(solver='liblinear', max_iter=1000)

# Εκπαίδευση του μοντέλου
logreg.fit(X, y)

# Εκτύπωση των συντελεστών
print("Coefficients:")
print(logreg.coef_)

# Προβλέψεις πιθανοτήτων
y_pred_probs = logreg.predict_proba(X)[:, 1]
print("Predicted Probabilities:")
print(y_pred_probs[:10])

# Κατωφλιοποίηση των προβλέψεων
threshold = 0.5
y_pred_class = np.where(y_pred_probs > threshold, 1, 0)

# Υπολογισμός του confusion matrix
conf_matrix = metrics.confusion_matrix(y, y_pred_class)
print("Confusion Matrix:")
print(conf_matrix)

# Υπολογισμός του classification report
classification_report = metrics.classification_report(y, y_pred_class)
print("Classification Report:")
print(classification_report)

# Υπολογισμός του training error rate
train_error_rate = 1 - logreg.score(X, y)
print(f"Training Error Rate: {train_error_rate}")

# Διαχωρισμός των δεδομένων σε training και test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)

# Εκπαίδευση του μοντέλου στα training data
logreg.fit(X_train, y_train)

# Προβλέψεις πιθανοτήτων για τα test data
y_pred_probs_test = logreg.predict_proba(X_test)[:, 1]

# Κατωφλιοποίηση των προβλέψεων για τα test data
y_pred_class_test = np.where(y_pred_probs_test > threshold, 1, 0)

# Υπολογισμός του confusion matrix για τα test data
conf_matrix_test = metrics.confusion_matrix(y_test, y_pred_class_test)
print("Confusion Matrix (Test Data):")
print(conf_matrix_test)



# Fitting Classification Trees

tree_mod = DecisionTreeClassifier(random_state=42)

data_encoded = pd.get_dummies(data, columns=['International plan', 'Voice mail plan', 'State'])

X = data_encoded.drop('Churn', axis=1)
y = data_encoded['Churn']
tree_mod.fit(X, y)

# Display summary
tree_rules = export_text(tree_mod, feature_names=list(X.columns))
print(tree_rules)

# Display the tree
export_graphviz(tree_mod, out_file='tree.dot', feature_names=list(X.columns),
                class_names=['False', 'True'], filled=True, rounded=True,
                special_characters=True)

# Calculate test classification
np.random.seed(2)
train_idx = np.random.choice(range(len(data)), int(len(data)/3), replace=False)
test_idx = np.setdiff1d(range(len(data)), train_idx)

data_test = data.iloc[test_idx]
churn_test = data['Churn'].iloc[test_idx]

tree_pred = tree_mod.predict(X.iloc[test_idx])
conf_matrix = confusion_matrix(churn_test, tree_pred)
print(conf_matrix)
accuracy = np.sum(np.diag(conf_matrix)) / np.sum(conf_matrix)
print(f'Accuracy: {accuracy}')

# Pruning the tree
np.random.seed(3)
tree_cv = DecisionTreeClassifier(random_state=42)
tree_cv.fit(X.iloc[train_idx], y.iloc[train_idx])

prune_path = tree_cv.cost_complexity_pruning_path(X.iloc[train_idx], y.iloc[train_idx])
alphas = prune_path.ccp_alphas

for alpha in alphas:
    pruned_tree = DecisionTreeClassifier(random_state=42, ccp_alpha=alpha)
    pruned_tree.fit(X.iloc[train_idx], y.iloc[train_idx])

    prune_pred = pruned_tree.predict(X.iloc[test_idx])
    prune_conf_matrix = confusion_matrix(churn_test, prune_pred)
    accuracy = np.sum(np.diag(prune_conf_matrix)) / np.sum(prune_conf_matrix)
    print(f'Alpha: {alpha}, Accuracy: {accuracy}')



#K-Nearest Neighbor Classification
print("\nK-Nearest Neighbors Classifier\n")
data_encoded = pd.get_dummies(data, columns=['International plan', 'Voice mail plan', 'State'])

X = data_encoded.drop('Churn', axis=1)
y = data_encoded['Churn']

random_state = int(time.time())
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X_train, y_train)

# Predictions on the test set
y_pred = neigh.predict(X_test)

# Print the predicted labels for the test set
print("Predicted Labels for the Test Set:")
print(y_pred)

# Compute accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2%}")

# Compute precision
precision = precision_score(y_test, y_pred)
print(f"Precision: {precision:.2%}")

#Random Forest Classification
print("\nRandom Forest Classifier\n")
data_encoded = pd.get_dummies(data, columns=['International plan', 'Voice mail plan', 'State'])

X = data_encoded.drop('Churn', axis=1)
y = data_encoded['Churn']

random_state = int(time.time())
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

# Initialize and train the Random Forest Classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)

# Predictions on the test set
y_pred = rf_classifier.predict(X_test)
print("Predicted Labels for the Test Set:")
print(y_pred)

# Compute accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2%}")

# Compute precision
precision = precision_score(y_test, y_pred)
print(f"Precision: {precision:.2%}")