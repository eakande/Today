from explainerdashboard import ExplainerDashboard, ClassifierExplainer, RegressionExplainer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
from explainerdashboard.custom import *
from sklearn.model_selection import RepeatedKFold
from dash_bootstrap_components.themes import FLATLY

import dash_bootstrap_components as dbc



#Importing Libraries & Packages


import openpyxl

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from pandas import DataFrame

#import portalocker
from dash import Dash, callback, html, dcc, dash_table, Input, Output, State, MATCH, ALL



#Import data
#############################################################################################
##### use the below data for both core and headline inflation

################### for Prices-more exogenous variable ################

#data = pd.read_excel('Data.xlsx', sheet_name="year").dropna()


#X = data.drop(['Headline Inflation', 'Core_farm', 'Core_farm_energy', 'Date'], axis=1)


######## switch y between core and headline

#y = data['Core_farm']

#y = data['Headline Inflation']
#############################################################################################


################### for others without extral CPI ################

# data = pd.read_excel('Data.xlsx', sheet_name="2001-nocpi").dropna() # drop all NAs

# data = data.loc[~(data==0).any(axis=1)]## drop all zeros

# X = data.drop(['Headline Inflation', 'Date'], axis=1)

# y = data['Headline Inflation']



##################################################
##### For Core Inflation 
######################################################


data = pd.read_excel('Data.xlsx', sheet_name="2001-nocpi").dropna()

data = data.loc[~(data==0).any(axis=1)]## drop all zeros

######## Core Inflation ####
X = data.drop(['Headline Inflation', 'Date'], axis=1)
y = data['Headline Inflation']


######## Core Inflation ####
#X = data.drop(['Headline Inflation', 'Food Prices', 'COP', 'Core_farm', 'Core_farm_energy','Imported_food','Date'], axis=1)
#y = data['Core_farm']


#Food.DataFrame(data.target,columns=["target"])


########## Dataset Split ########


X_train = X[X.index < 90]
y_train = y[y.index < 90]              
    
X_test = X[X.index >= 90]    
y_test = y[y.index >= 90]


# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# print(X_train.shape,y_train.shape,X_test.shape,y_test.shape)



model = RandomForestRegressor(n_estimators = 400,
                           n_jobs = -1,
                           oob_score = True,
                           bootstrap = True,
                            max_depth=5,
                           random_state = 42)



model.fit(X_train, y_train.values.ravel())


model.score (X_train,y_train), model.score(X_test,y_test),model.oob_score_


#X_train, y_train, X_test, y_test = titanic_survive()
#train_names, test_names = titanic_names()
#model = RandomForestClassifier(n_estimators=50, max_depth=5)
#model.fit(X_train, y_train)




explainer = RegressionExplainer(model, X, y)


#ExplainerDashboard(explainer).run()


class CustomModelTab(ExplainerComponent):
    def __init__(self, explainer, name=None):
        super().__init__(explainer, title="Selected Drivers")
        self.importance =  ImportancesComposite(explainer,
                                title='Impact',
                                hide_importances=False)
        self.register_components()

    def layout(self):
        return dbc.Container([
            dbc.Row([
                dbc.Col([
                    self.importance.layout(),
                    html.H3(f"Clearly {self.explainer.columns_ranked_by_shap()[0]} was the most important"
                            f", followed by {self.explainer.columns_ranked_by_shap()[1]}"
                            f" and {self.explainer.columns_ranked_by_shap()[2]}.")
                ])
            ])
        ])




class CustomModelTab1(ExplainerComponent):
    def __init__(self, explainer, name=None):
        super().__init__(explainer, title="Model Performance")
        self.Reg_summary = RegressionModelStatsComposite(explainer,
                                title='Impact',
                                hide_predsvsactual=False, hide_residuals=False,
                                hide_regvscol=False)
        self.register_components()

    def layout(self):
           return dbc.Container([
               dbc.Row([
                   dbc.Col([
                       self.Reg_summary.layout(),
                      
                      
                   ])
               ])
           ])
    

class CustomPredictionsTab(ExplainerComponent):
    def __init__(self, explainer, name=None):
        super().__init__(explainer, title="Model Predictions")

        self.prediction = IndividualPredictionsComposite(explainer,
                                                    hide_predindexselector=False, hide_predictionsummary=False,
                                                    hide_contributiongraph=False, hide_pdp=False,
                                                    hide_contributiontable=False)
        self.register_components()

    def layout(self):
        return dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H3("Individual Prediction:"),
                    self.prediction.layout()
                ])
                
            ])
        ])
    


class CustomPredictionsTab2(ExplainerComponent):
    def __init__(self, explainer, name=None):
        super().__init__(explainer, title="What if Scenarios")

        self.what_if = WhatIfComposite(explainer,
                                                    hide_whatifindexselector=False, hide_inputeditor=False,
                                                    hide_whatifcontribution=False, hide_whatifpdp=False)
        self.register_components()

    def layout(self):
        return dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H3("Individual Prediction:"),
                    self.what_if.layout()
                ])
                
            ])
        ])
    
    
    
    

class CustomPredictionsTab3(ExplainerComponent):
    def __init__(self, explainer, name=None):
        super().__init__(explainer, title="SHAP Dependencies")

        self.shap_depend = ShapDependenceComposite(explainer,
                                                    hide_shapsummary=False, hide_shapdependence=False)
        self.register_components()

    def layout(self):
        return dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H3("SHAP Dependencies:"),
                    self.shap_depend.layout()
                ])
                
            ])
        ])
    
    
    
class CustomPredictionsTab4(ExplainerComponent):
    def __init__(self, explainer):
        super().__init__(explainer, title="Interacting Features")

        self.interaction = ShapInteractionsComposite(explainer, 
                                                      hide_interactionsummary=False, 
                                                      hide_interactiondependence=False)
        self.register_components()

    def layout(self):
        return dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H3("Interacting Features:"),
                    self.interaction.layout()
                ])

            ])
        ])

from dash_bootstrap_components.themes import CYBORG
#from dash_bootstrap_components.themes import LUMEN

db=ExplainerDashboard(explainer, [CustomModelTab, CustomModelTab1, CustomPredictionsTab,
                               CustomPredictionsTab2, CustomPredictionsTab3, CustomPredictionsTab4], 
                        title='Inflation Explainer Dashboard for Nigeria', header_hide_selector=False,
                        bootstrap=CYBORG)


db.to_yaml("dashboard.yaml", explainerfile="explainer.joblib", dump_explainer=True)

db = ExplainerDashboard.from_config("dashboard.yaml")
app = db.app
server = db.flask_server()
