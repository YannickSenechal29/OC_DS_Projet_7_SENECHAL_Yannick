import pickle
import flask
import pandas as pd

# Load our model with pickle (adresses may be changed for serialisation)
model = pickle.load(open('API/lgbm_home_risk_model.pkl', 'rb'))
# Load data test sample (index_col = 0 to keep id of loan)
df_test_sample = pd.read_csv('API/test_sample_data_home_risk.csv', index_col = 0)
# Drop predict last column predict proba of our dataframe not used here
df_test_sample.drop(columns = 'TARGET_PROB', inplace = True)
# Define the treshold of our application for refused loan
treshold = 0.49
# defining flask pages
app = flask.Flask(__name__)
app.config["DEBUG"] = True
# definig home page
@app.route('/', methods=['GET'])
def home():
    return "<h1>My first Flask API</h1><p>This site is a prototype API \
    for home risk project 7 of OpenClassRooms DataScientist training.</p>"
# defining page for the results of a prediction via index
@app.route('/scores', methods=['GET'])
def predict():
    # get the index from a request
    data_index = flask.request.args.get('index')
    # get inputs features from the data with index. Carreful you must pass data_index in int!! Last column must be drop when calling the model
    input = df_test_sample.loc[int(data_index), :]
    # predict probability score. Carreful you must reshape your input data since we have only one sample!!
    model_prediction = model.predict_proba(input.values.reshape(1,-1))
    # create a dictionnary to janosify. We get the probability to be positive and evaluate if the credit is accepted (1) or denied (0)!
    # careful bool object must be transformed in int for JSON format!
    dict_result = {'ID_loan':int(data_index), 'Credit_score': model_prediction[:,1][0], 'Answer': int(model_prediction[:,1][0] < treshold)}
    return flask.jsonify(dict_result)
# define endpoint for Flask
app.add_url_rule('/scores', 'scores', predict)

app.run()
