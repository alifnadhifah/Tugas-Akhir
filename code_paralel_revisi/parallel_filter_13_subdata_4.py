import warnings
warnings.filterwarnings('ignore')
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import linear_kernel
from sklearn.metrics.pairwise import cosine_similarity
# from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
# from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import ndcg_score, recall_score, precision_score

from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split as surprise_train_test_split
from surprise import accuracy

# Tentukan path direktori Drive
base_item_path = "E:/TA ALIF/Revisi/sub-data_final/"
sub_data_item_path_4 =  f"{base_item_path}sub-data_item_final_4.csv"
sub_data_item_4 =  pd.read_csv(sub_data_item_path_4)

base_user_path = "E:/TA ALIF/Revisi/sub-data_final/"
sub_data_user_path_4 =  f"{base_user_path}sub-data_user_final_4.csv"
sub_data_user_4 =  pd.read_csv(sub_data_user_path_4)

# Tentukan path direktori Drive
base_rating_path = "E:/TA ALIF/Revisi/sub-data_final/"
sub_data_rating_path_4 =  f"{base_rating_path}sub-data_rating_final_4.csv"
sub_data_rating_4 =  pd.read_csv(sub_data_rating_path_4)

# Tentukan path direktori Drive
base_cluster_path = "E:/TA ALIF/Revisi/sub-data_final/"
sub_data_cluster_path_4 =  f"{base_cluster_path}sub-data_cluster_final_4.csv"
sub_data_cluster_4 =  pd.read_csv(sub_data_cluster_path_4)

## TF-IDF SUB DATA 1
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')
nltk.download('stopwords')

# Membuat daftar stopwords
stop_words = set(stopwords.words('english'))

# Fungsi untuk membersihkan dan memproses teks
def preprocess(text):
    # Handle non-string values
    if not isinstance(text, str):
        return ''  # Return an empty string for non-string values
    # Tokenisasi
    tokens = word_tokenize(text)
    # Menghapus stopwords dan simbol
    tokens = [word for word in tokens if word.isalnum() and word.lower() not in stop_words]
    # Gabungkan kembali menjadi string
    return ' '.join(tokens)

# Preroses synopsis
sub_data_cluster_4['cleaned_synopsis'] = sub_data_cluster_4['Anime_Synopsis'].apply(preprocess)
# Misalkan id1 dan id2 adalah ID item yang ingin dicetak
id1 = 0
id2 = 1

# Menampilkan sinopsis dan cleaned sinopsis untuk item dengan ID tertentu
print(f"Item ID {id1}:")
print("Anime Synopsis:", sub_data_cluster_4.loc[id1, 'Anime_Synopsis'])
print("Cleaned Synopsis:", sub_data_cluster_4.loc[id1, 'cleaned_synopsis'])
print()

print(f"Item ID {id2}:")
print("Anime Synopsis:", sub_data_cluster_4.loc[id2, 'Anime_Synopsis'])
print("Cleaned Synopsis:", sub_data_cluster_4.loc[id2, 'cleaned_synopsis'])

# Menggunakan TfidfVectorizer untuk menghitung TF-IDF
tfidf_4 = TfidfVectorizer(stop_words='english')
tfidf_matrix_4 = tfidf_4.fit_transform(sub_data_cluster_4['cleaned_synopsis'])
tfidf_matrix_normalized_4 = normalize(tfidf_matrix_4)

# Menghitung kemiripan kosinus
cosine_sim_4 = linear_kernel(tfidf_matrix_4, tfidf_matrix_4)

# Misalkan 'anime_id' adalah index dari df
anime_ids_4 = sub_data_cluster_4.index

# Membuat DataFrame dari cosine similarity
cosine_sim_df_4 = pd.DataFrame(cosine_sim_4, index=anime_ids_4, columns=anime_ids_4)


## ITEM BASED FILTERING SUB DATA 1
# Convert the DataFrame into Surprise's format
reader_4 = Reader(rating_scale=(sub_data_rating_4['Rating_by_User'].min(), sub_data_rating_4['Rating_by_User'].max()))
data_4 = Dataset.load_from_df(sub_data_rating_4[['User_Id', 'Anime_Id', 'Rating_by_User']], reader_4)

# Split the data into training and test sets
trainset_4, testset_4 = surprise_train_test_split(data_4, test_size=0.2, random_state=42)

# Instantiate the SVD model
svd_4 = SVD()

# Train the SVD model
svd_4.fit(trainset_4)

# Predict ratings for the testset
predictions_4 = svd_4.test(testset_4)

# Mengonversi predictions ke DataFrame
predictions_df_4 = pd.DataFrame(predictions_4, columns=['User_Id', 'Anime_Id', 'Rating_by_User', 'Predicted_Rating', 'Details'])

# Menampilkan tabel predictions
predictions_df_4.head()

# FILTER 1-3
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import time

class HybridRecommender2:
    def __init__(self, data_item, data_rating, svd_model, recommendation_folder):
        self.data_item = data_item
        self.data_rating = data_rating
        self.svd_model = svd_model
        self.recommendation_folder = recommendation_folder

    def get_items_in_cluster(self, item_id):
        query_cluster = self.data_item.loc[self.data_item['Anime_Id'] == item_id, 'Cluster'].values[0]
        cluster_items = self.data_item.loc[self.data_item['Cluster'] == query_cluster]
        return cluster_items.drop(cluster_items[cluster_items['Anime_Id'] == item_id].index).reset_index(drop=True)

    def predict_ratings_for_items(self, user_id, item_ids):
        predictions = []
        for item_id in item_ids:
            prediction = self.svd_model.predict(user_id, item_id)
            predictions.append({
                'User_Id': user_id,
                'Anime_Id': item_id,
                'Predicted_Rating': prediction.est
            })
        return pd.DataFrame(predictions)

    def get_top_N_recommendations(self, user_id, item_id, N):
        items_in_cluster = self.get_items_in_cluster(item_id)
        list_anime_id_in_cluster = items_in_cluster['Anime_Id'].tolist()
        predicted_ratings_df = self.predict_ratings_for_items(user_id, list_anime_id_in_cluster)
        final_predicted_ratings_df = pd.merge(predicted_ratings_df, self.data_rating, on=['User_Id', 'Anime_Id'], how='left')
        top_N_rating_prediction = final_predicted_ratings_df.sort_values(by='Predicted_Rating', ascending=False).head(N)
        list_anime_id_top_N_rating_prediction = top_N_rating_prediction['Anime_Id'].tolist()
        filtered_df = self.data_item[self.data_item['Anime_Id'].isin(list_anime_id_top_N_rating_prediction)]
        top_N_rating_prediction['Anime_Title'] = filtered_df['Anime_Title'].tolist()
        top_N_rating_prediction['Anime_Genre'] = filtered_df['Anime_Genre'].tolist()
        return top_N_rating_prediction.sort_values(by='Predicted_Rating', ascending=False).reset_index(drop=True)

    def calculate_metrics(self, recommendations, actual_ratings, n):
        relevant_items = actual_ratings[actual_ratings['Rating_by_User'] >= 7]['Anime_Id'].tolist()
        recommended_items = recommendations['Anime_Id'].tolist()[:n]

        tp = len(set(recommended_items) & set(relevant_items))
        fp = len(set(recommended_items) - set(relevant_items))
        fn = len(set(relevant_items) - set(recommended_items))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f2_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        return precision, recall, f2_score

    def evaluate_model(self, n):
        total_precision = 0
        total_recall = 0
        total_f2_score = 0
        count = 0

        # Loop through all recommendation files
        for file_name in os.listdir(self.recommendation_folder):
            if file_name.endswith('.csv'):
                file_path = os.path.join(self.recommendation_folder, file_name)
                recommendations = pd.read_csv(file_path)

                user_id = recommendations['User_Id'].iloc[0]
                actual_ratings = self.data_rating[self.data_rating['User_Id'] == user_id]

                for item_id in recommendations['Query_Anime_Id'].unique():
                    item_recommendations = recommendations[recommendations['Query_Anime_Id'] == item_id]
                    precision, recall, f2_score = self.calculate_metrics(item_recommendations, actual_ratings, n)

                    total_precision += precision
                    total_recall += recall
                    total_f2_score += f2_score
                    count += 1

        avg_precision_4 = total_precision / count
        avg_recall_4 = total_recall / count
        avg_f2_score_4 = total_f2_score / count

        return avg_precision_4, avg_recall_4, avg_f2_score_4
        
recommendation_folder2_sd4 = 'E:/TA ALIF/Revisi/new_get_recommendations_r2/sub-data_4'  # Ganti dengan path folder yang sesuai

# Create the directory if it doesn't exist
os.makedirs(recommendation_folder2_sd4, exist_ok=True)

recommender2_sd4 = HybridRecommender2(sub_data_cluster_4, sub_data_rating_4, svd_4, recommendation_folder2_sd4)

import pandas as pd
import os
import time

# Loop through all users and all items
def getUserRecommendation(user_id):
    file_name = f'user_{user_id}_recommendations.csv'
    file_path = os.path.join(recommendation_folder2_sd4, file_name)

    # Check if the file already exists
    if os.path.exists(file_path):
        # print(f"File for User {user_id} already exists. Skipping.")
        return

    start_time = time.time()  # Start time
    # print(f"{user_id} in progress")

    user_recommendations = []  # List to hold recommendations for the current user

    for item_id in sub_data_cluster_4['Anime_Id']:
        top_10_recommendations = recommender2_sd4.get_top_N_recommendations(user_id=user_id, item_id=item_id, N=10)

        # Ensure the recommendations are in a DataFrame format
        if isinstance(top_10_recommendations, list):
            top_10_recommendations_df = pd.DataFrame(top_10_recommendations)
        else:
            top_10_recommendations_df = top_10_recommendations

        # Add user_id and item_id columns to the DataFrame
        top_10_recommendations_df['User_Id'] = user_id
        top_10_recommendations_df['Query_Anime_Id'] = item_id

        # Append to the list
        user_recommendations.append(top_10_recommendations_df)

    # Concatenate all DataFrames for the current user into a single DataFrame
    user_recommendations_df = pd.concat(user_recommendations, ignore_index=True)

    # Save the DataFrame to a CSV file named based on the user_id
    user_recommendations_df.to_csv(file_path, index=False)

    end_time = time.time()  # End time
    duration = end_time - start_time  # Calculate the duration
    # print(f"User {user_id} completed in {duration:.2f} seconds. Saved to {file_path}")

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

data_user = sub_data_user_4
total_proses = int(input("Total Process: "))
user_data_chunks = list(chunks(data_user['User_Id'], len(data_user['User_Id'])//total_proses))
num_process = int(input("Process ke : "))

for user_id in tqdm(user_data_chunks[num_process]):
    getUserRecommendation(user_id)

# print("EVALUASI MODEL")
# avg_precision2_sd4, avg_recall2_sd4, avg_f2_score2_sd4 = recommender2_sd4.evaluate_model(n=10)
# print(f"Avg Precision: {avg_precision2_sd4}, Avg Recall: {avg_recall2_sd4}, Avg F1 Score: {avg_f2_score2_sd4}")
# output_eval = "E:/TA ALIF/Revisi/hasil_evaluasi_baru/sub-data_4/hasil_eval_r2.txt"
# # Create the directory if it doesn't exist
# os.makedirs(os.path.dirname(output_eval), exist_ok=True)
# with open(output_eval, 'w') as f:
#     f.write(f"Avg Precision: {avg_precision2_sd4}, Avg Recall: {avg_recall2_sd4}, Avg F1 Score: {avg_f2_score2_sd4}")