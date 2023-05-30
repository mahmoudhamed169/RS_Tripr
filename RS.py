from flask import Flask,request,jsonify
import numpy as np
import scipy
import pandas as pd
import math
import random
import sklearn
from nltk.corpus import stopwords
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse.linalg import svds
from sklearn.preprocessing import MinMaxScaler

app=Flask(__name__)
@app.route("/api",methods=["POST"])
def predict():
    if request.method == "POST":
        if 'interactions' not in request.files:
            return 'No file uploaded.'
        if 'id' not in request.form:
            return 'No id.'

        file = request.files['interactions']
        id = int(request.form.get('id'))

        # Load the place data
        places_df = pd.read_excel('place_2.xlsx')
        interactions_df = pd.read_csv(file)

        event_type_strength = {
            'VIEW': 1.0,
            'LIKE': 2.0,
            'BOOKMARK': 2.5,
            'FOLLOW': 3.0,
            'COMMENT CREATED': 4.0,
        }

        interactions_df['eventStrength'] = interactions_df['eventType'].apply(lambda x: event_type_strength[x])

        users_interactions_count_df = interactions_df.groupby(['personId', 'placeId']).size().groupby('personId').size()

        users_with_enough_interactions_df = users_interactions_count_df[users_interactions_count_df >= 5].reset_index()[
            ['personId']]

        interactions_from_selected_users_df = interactions_df.merge(users_with_enough_interactions_df,
                                                                    how='right',
                                                                    left_on='personId',
                                                                    right_on='personId')

        def smooth_user_preference(x):
            return math.log(1 + x, 2)

        interactions_full_df = interactions_from_selected_users_df \
            .groupby(['personId', 'placeId'])['eventStrength'].sum() \
            .apply(smooth_user_preference).reset_index()

        interactions_train_df, interactions_test_df = train_test_split(interactions_full_df,
                                                                       stratify=interactions_full_df['personId'],
                                                                       test_size=0.20,
                                                                       random_state=42)

        interactions_full_indexed_df = interactions_full_df.set_index('personId')
        interactions_train_indexed_df = interactions_train_df.set_index('personId')
        interactions_test_indexed_df = interactions_test_df.set_index('personId')

        def get_items_interacted(person_id, interactions_df):
            # Get the user's data and merge in the movie information.
            interacted_items = interactions_df.loc[person_id]['placeId']
            return set(interacted_items if type(interacted_items) == pd.Series else [interacted_items])

        # Top-N accuracy metrics consts
        EVAL_RANDOM_SAMPLE_NON_INTERACTED_ITEMS = 100

        item_popularity_df = interactions_full_df.groupby('placeId')['eventStrength'].sum().sort_values(
            ascending=False).reset_index()

        import nltk
        nltk.download('stopwords')

        # Ignoring stopwords (words with no semantics) from English and Portuguese (as we have a corpus with mixed languages)
        stopwords_list = stopwords.words('english') + stopwords.words('portuguese')

        # Trains a model whose vectors size is 5000, composed by the main unigrams and bigrams found in the corpus, ignoring stopwords
        vectorizer = TfidfVectorizer(analyzer='word',
                                     ngram_range=(1, 2),
                                     min_df=0.003,
                                     max_df=0.5,
                                     max_features=5000,
                                     stop_words=stopwords_list)

        item_ids = places_df['placeId'].tolist()
        tfidf_matrix = vectorizer.fit_transform(places_df['category'])
        tfidf_feature_names = vectorizer.vocabulary_.keys()

        tfidf_matrix

        def get_item_profile(item_id):
            idx = item_ids.index(item_id)
            item_profile = tfidf_matrix[idx:idx + 1]
            return item_profile

        def get_item_profiles(ids):
            item_profiles_list = [get_item_profile(x) for x in ids]
            item_profiles = scipy.sparse.vstack(item_profiles_list)
            return item_profiles

        def build_users_profile(person_id, interactions_indexed_df):
            interactions_person_df = interactions_indexed_df.loc[person_id]
            user_item_profiles = get_item_profiles(interactions_person_df['placeId'])

            user_item_strengths = np.array(interactions_person_df['eventStrength']).reshape(-1, 1)
            # Weighted average of item profiles by the interactions strength
            user_item_strengths_weighted_avg = np.sum(user_item_profiles.multiply(user_item_strengths),
                                                      axis=0) / np.sum(
                user_item_strengths)

            user_item_strengths_weighted_avg_arr = np.asarray(
                user_item_strengths_weighted_avg)  # Convert to numpy array

            user_profile_norm = sklearn.preprocessing.normalize(user_item_strengths_weighted_avg_arr)
            return user_profile_norm

        def build_users_profiles():
            interactions_indexed_df = interactions_train_df[interactions_train_df['placeId'] \
                .isin(places_df['placeId'])].set_index('personId')
            user_profiles = {}
            for person_id in interactions_indexed_df.index.unique():
                user_profiles[person_id] = build_users_profile(person_id, interactions_indexed_df)
            return user_profiles

        user_profiles = build_users_profiles()
        myprofile = user_profiles[2425]
        pd.DataFrame(sorted(zip(tfidf_feature_names,
                                user_profiles[2425].flatten().tolist()), key=lambda x: -x[1])[:20],
                     columns=['token', 'relevance'])

        class ModelEvaluator:

            def get_not_interacted_items_sample(self, person_id, sample_size, seed=42):
                interacted_items = get_items_interacted(person_id, interactions_full_indexed_df)
                all_items = set(places_df['placeId'])
                non_interacted_items = all_items - interacted_items

                random.seed(seed)
                non_interacted_items_sample = random.sample(non_interacted_items, sample_size)
                return set(non_interacted_items_sample)

            def _verify_hit_top_n(self, item_id, recommended_items, topn):
                try:
                    index = next(i for i, c in enumerate(recommended_items) if c == item_id)
                except:
                    index = -1
                hit = int(index in range(0, topn))
                return hit, index

            def evaluate_model_for_user(self, model, person_id):
                # Getting the items in test set
                interacted_values_testset = interactions_test_indexed_df.loc[person_id]
                if type(interacted_values_testset['placeId']) == pd.Series:
                    person_interacted_items_testset = set(interacted_values_testset['placeId'])
                else:
                    person_interacted_items_testset = set([int(interacted_values_testset['placeId'])])
                interacted_items_count_testset = len(person_interacted_items_testset)

                # Getting a ranked recommendation list from a model for a given user
                person_recs_df = model.recommend_items(person_id,
                                                       items_to_ignore=get_items_interacted(person_id,
                                                                                            interactions_train_indexed_df),
                                                       topn=10000000000)

                hits_at_5_count = 0
                hits_at_10_count = 0
                # For each item the user has interacted in test set
                for item_id in person_interacted_items_testset:
                    # Getting a random sample (100) items the user has not interacted
                    # (to represent items that are assumed to be no relevant to the user)
                    non_interacted_items_sample = self.get_not_interacted_items_sample(person_id,
                                                                                       sample_size=EVAL_RANDOM_SAMPLE_NON_INTERACTED_ITEMS,
                                                                                       seed=item_id % (2 ** 32))
                    # Combining the current interacted item with the 100 random items
                    items_to_filter_recs = non_interacted_items_sample.union(set([item_id]))

                    # Filtering only recommendations that are either the interacted item or from a random sample of 100 non-interacted items
                    valid_recs_df = person_recs_df[person_recs_df['placeId'].isin(items_to_filter_recs)]
                    valid_recs = valid_recs_df['placeId'].values
                    # Verifying if the current interacted item is among the Top-N recommended items
                    hit_at_5, index_at_5 = self._verify_hit_top_n(item_id, valid_recs, 5)
                    hits_at_5_count += hit_at_5
                    hit_at_10, index_at_10 = self._verify_hit_top_n(item_id, valid_recs, 10)
                    hits_at_10_count += hit_at_10

                # Recall is the rate of the interacted items that are ranked among the Top-N recommended items,
                # when mixed with a set of non-relevant items
                recall_at_5 = hits_at_5_count / float(interacted_items_count_testset)
                recall_at_10 = hits_at_10_count / float(interacted_items_count_testset)

                person_metrics = {'hits@5_count': hits_at_5_count,
                                  'hits@10_count': hits_at_10_count,
                                  'interacted_count': interacted_items_count_testset,
                                  'recall@5': recall_at_5,
                                  'recall@10': recall_at_10}
                return person_metrics

            def evaluate_model(self, model):
                # print('Running evaluation for users')
                people_metrics = []
                for idx, person_id in enumerate(list(interactions_test_indexed_df.index.unique().values)):
                    # if idx % 100 == 0 and idx > 0:
                    #    print('%d users processed' % idx)
                    person_metrics = self.evaluate_model_for_user(model, person_id)
                    person_metrics['_person_id'] = person_id
                    people_metrics.append(person_metrics)
                print('%d users processed' % idx)

                detailed_results_df = pd.DataFrame(people_metrics) \
                    .sort_values('interacted_count', ascending=False)

                global_recall_at_5 = detailed_results_df['hits@5_count'].sum() / float(
                    detailed_results_df['interacted_count'].sum())
                global_recall_at_10 = detailed_results_df['hits@10_count'].sum() / float(
                    detailed_results_df['interacted_count'].sum())

                global_metrics = {'modelName': model.get_model_name(),
                                  'recall@5': global_recall_at_5,
                                  'recall@10': global_recall_at_10}
                return global_metrics, detailed_results_df

        model_evaluator = ModelEvaluator()

        class ContentBasedRecommender:
            MODEL_NAME = 'Content-Based'

            def __init__(self, items_df=None):
                self.item_ids = item_ids
                self.items_df = items_df

            def get_model_name(self):
                return self.MODEL_NAME

            def _get_similar_items_to_user_profile(self, person_id, topn=1000):
                # Computes the cosine similarity between the user profile and all item profiles
                cosine_similarities = cosine_similarity(user_profiles[person_id], tfidf_matrix)
                # Gets the top similar items
                similar_indices = cosine_similarities.argsort().flatten()[-topn:]
                # Sort the similar items by similarity
                similar_items = sorted([(item_ids[i], cosine_similarities[0, i]) for i in similar_indices],
                                       key=lambda x: -x[1])
                return similar_items

            def recommend_items(self, user_id, items_to_ignore=[], topn=10, verbose=False):
                similar_items = self._get_similar_items_to_user_profile(user_id)
                # Ignores items the user has already interacted
                similar_items_filtered = list(filter(lambda x: x[0] not in items_to_ignore, similar_items))

                recommendations_df = pd.DataFrame(similar_items_filtered, columns=['placeId', 'recStrength']) \
                    .head(topn)

                if verbose:
                    if self.items_df is None:
                        raise Exception('"items_df" is required in verbose mode')

                    recommendations_df = recommendations_df.merge(self.items_df, how='left',
                                                                  left_on='placeId',
                                                                  right_on='placeId')[
                        ['recStrength', 'Name', 'placeId', 'category']]

                return recommendations_df

        content_based_recommender_model = ContentBasedRecommender(places_df)

        users_items_pivot_matrix_df = interactions_train_df.pivot(index='personId',
                                                                  columns='placeId',
                                                                  values='eventStrength').fillna(0)

        users_items_pivot_matrix_df.head(10)

        users_items_pivot_matrix = users_items_pivot_matrix_df.to_numpy()
        users_items_pivot_matrix[:10]

        users_ids = list(users_items_pivot_matrix_df.index)
        users_ids[:10]

        users_items_pivot_sparse_matrix = csr_matrix(users_items_pivot_matrix)
        users_items_pivot_sparse_matrix

        # The number of factors to factor the user-item matrix.
        NUMBER_OF_FACTORS_MF = 15
        # Performs matrix factorization of the original user item matrix
        # U, sigma, Vt = svds(users_items_pivot_matrix, k = NUMBER_OF_FACTORS_MF)
        U, sigma, Vt = svds(users_items_pivot_sparse_matrix, k=NUMBER_OF_FACTORS_MF)

        sigma = np.diag(sigma)
        sigma.shape

        all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt)
        all_user_predicted_ratings

        all_user_predicted_ratings_norm = (all_user_predicted_ratings - all_user_predicted_ratings.min()) / (
                all_user_predicted_ratings.max() - all_user_predicted_ratings.min())

        # Converting the reconstructed matrix back to a Pandas dataframe
        cf_preds_df = pd.DataFrame(all_user_predicted_ratings_norm, columns=users_items_pivot_matrix_df.columns,
                                   index=users_ids).transpose()

        class CFRecommender:
            MODEL_NAME = 'Collaborative Filtering'

            def __init__(self, cf_predictions_df, items_df=None):
                self.cf_predictions_df = cf_predictions_df
                self.items_df = items_df

            def get_model_name(self):
                return self.MODEL_NAME

            def recommend_items(self, user_id, items_to_ignore=[], topn=10, verbose=False):
                # Get and sort the user's predictions
                sorted_user_predictions = self.cf_predictions_df[user_id].sort_values(ascending=False) \
                    .reset_index().rename(columns={user_id: 'recStrength'})

                # Recommend the highest predicted rating movies that the user hasn't seen yet.
                recommendations_df = sorted_user_predictions[~sorted_user_predictions['placeId'].isin(items_to_ignore)] \
                    .sort_values('recStrength', ascending=False) \
                    .head(topn)

                if verbose:
                    if self.items_df is None:
                        raise Exception('"items_df" is required in verbose mode')

                    recommendations_df = recommendations_df.merge(self.items_df, how='left',
                                                                  left_on='placeId',
                                                                  right_on='placeId')[
                        ['recStrength', 'Name', 'placeId', 'category']]
                return recommendations_df

        cf_recommender_model = CFRecommender(cf_preds_df, places_df)

        class HybridRecommender:
            MODEL_NAME = 'Hybrid'

            def __init__(self, cb_rec_model, cf_rec_model, items_df, cb_ensemble_weight=1.0, cf_ensemble_weight=1.0):
                self.cb_rec_model = cb_rec_model
                self.cf_rec_model = cf_rec_model
                self.cb_ensemble_weight = cb_ensemble_weight
                self.cf_ensemble_weight = cf_ensemble_weight
                self.items_df = items_df

            def get_model_name(self):
                return self.MODEL_NAME

            def recommend_items(self, user_id, items_to_ignore=[], topn=10, verbose=False):
                # Getting the top-1000 Content-based filtering recommendations
                cb_recs_df = self.cb_rec_model.recommend_items(user_id, items_to_ignore=items_to_ignore,
                                                               verbose=verbose,
                                                               topn=1000).rename(
                    columns={'recStrength': 'recStrengthCB'})

                # Getting the top-1000 Collaborative filtering recommendations
                cf_recs_df = self.cf_rec_model.recommend_items(user_id, items_to_ignore=items_to_ignore,
                                                               verbose=verbose,
                                                               topn=1000).rename(
                    columns={'recStrength': 'recStrengthCF'})

                # Combining the results by contentId
                recs_df = cb_recs_df.merge(cf_recs_df,
                                           how='outer',
                                           left_on='placeId',
                                           right_on='placeId').fillna(0.0)

                # Computing a hybrid recommendation score based on CF and CB scores
                # recs_df['recStrengthHybrid'] = recs_df['recStrengthCB'] * recs_df['recStrengthCF']
                recs_df['recStrengthHybrid'] = (recs_df['recStrengthCB'] * self.cb_ensemble_weight) \
                                               + (recs_df['recStrengthCF'] * self.cf_ensemble_weight)

                # Sorting recommendations by hybrid score
                recommendations_df = recs_df.sort_values('recStrengthHybrid', ascending=False).head(topn)

                if verbose:
                    if self.items_df is None:
                        raise Exception('"items_df" is required in verbose mode')

                    recommendations_df = recommendations_df.merge(self.items_df, how='left',
                                                                  left_on='placeId',
                                                                  right_on='placeId')[
                        ['recStrengthHybrid', 'placeId', 'Name', 'category']]

                return recommendations_df

        hybrid_recommender_model = HybridRecommender(content_based_recommender_model, cf_recommender_model, places_df,
                                                     cb_ensemble_weight=1.0, cf_ensemble_weight=100.0)

        prediction = hybrid_recommender_model.recommend_items(id, topn=20, verbose=True)
        print(prediction)
        dic_df = prediction.to_dict()
        print(dic_df)
        return jsonify(dic_df["placeId"])


if __name__ == '__main__' :
    app.run(debug=True)