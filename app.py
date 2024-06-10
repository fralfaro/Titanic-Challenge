import streamlit as st
import base64
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
import plotly.figure_factory as ff
import altair as alt


from itables.streamlit import interactive_table
from great_tables import GT, html


# Initial page config
st.set_page_config(
    page_title="Titanic - ML from Disaster",
    layout="wide",
    initial_sidebar_state="expanded",
)


# extra classes
class SidebarText:
    introduction = """
        <small> The [RMS Titanic](https://es.wikipedia.org/wiki/RMS_Titanic) 
        as a British passenger
        liner that tragically sank in the North Atlantic
        Ocean on April 15, 1912, during its maiden voyage
        from Southampton, England, to New York City. It
        was the largest ship afloat at the time and was considered to be unsinkable, but it struck an iceberg 
        and went down, resulting in a significant loss of life.</small>
        """
    goals = """
        <small>  This project aims to unravel
        the hidden patterns and unveil insights
        surrounding the tragic sinking of the RMS Titanic by harnessing
        the power of machine learning and a user-friendly
        web application framework called Streamlit. </small> 
        """

class BodyText:
    eda_intro = """
    The Titanic dataset is one of the most well-known datasets in the field of data science and machine learning. It contains detailed information about the passengers aboard the Titanic, a British passenger liner that tragically sank in the North Atlantic Ocean on April 15, 1912, after hitting an iceberg. This disaster resulted in the loss of over 1,500 lives and has since become a poignant example of maritime tragedy.

    Exploratory Data Analysis (EDA) is a crucial step in any data analysis project. It involves examining the dataset to uncover underlying patterns, spot anomalies, test hypotheses, and check assumptions through summary statistics and graphical representations. For the Titanic dataset, EDA helps us understand the factors that influenced survival rates, such as passenger demographics, socio-economic status, and travel details.

    The dataset comprises variables such as passenger age, gender, ticket class, fare paid, and whether or not the passenger survived. By analyzing these variables, we can gain insights into which groups of passengers were more likely to survive and the reasons behind these trends. For instance, we might explore questions like:

    - Did gender play a significant role in survival rates?
    - Were first-class passengers more likely to survive than those in lower classes?
    - How did the age of passengers affect their chances of survival?

    Through various visualizations and statistical analyses, EDA provides a foundation for more complex modeling and predictive analysis. It allows us to clean and preprocess the data, handle missing values, and create new features that might improve the performance of machine learning models.
    """
    fe_intro = """
    Feature engineering is a crucial step in the data preprocessing pipeline, aimed at enhancing the predictive power of machine learning models. For the Titanic dataset, this involves creating new features and modifying existing ones to better capture the underlying patterns that influence passenger survival.

    The Titanic dataset includes various columns such as 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', and 'Embarked'. Each of these features holds potential insights into the survival outcomes, but they often require transformation and enrichment to become more effective for predictive modeling.

    Feature engineering is an iterative process that involves experimenting with different transformations and evaluating their impact on model performance. By carefully crafting and selecting features, we can significantly improve the accuracy and robustness of predictive models for the Titanic dataset.
    """
    fe_conclusion = """
    In our feature engineering process for the Titanic dataset, we undertook several steps to prepare the data for effective modeling:

    1. **Removal of Non-Contributory Columns:** We removed the 'Name' and 'Ticket' columns, as they did not provide significant predictive value for our model.
    2. **Handling Missing Values:**
       - For the 'Age' column, missing values were filled with the mean age to maintain consistency and avoid data loss.
       - For the 'Cabin' column, missing values were replaced with the most frequent value ('N'), and only the first letter of the cabin was retained to simplify the data.
    3. **Data Type Conversion:** The columns 'Pclass', 'SibSp', and 'Parch' were converted from numerical to string type to better capture categorical relationships.
    4. **Data Saving:** The processed training and test datasets were saved for future modeling and analysis.

    These feature engineering steps have improved the quality and usability of the dataset, ensuring that it is well-prepared for subsequent analysis and machine learning tasks. By addressing missing values, simplifying categorical data, and removing unnecessary columns, we have created a more robust and interpretable dataset for predicting passenger survival on the Titanic.
    """
    mle_intro = """
    The Titanic dataset is a popular and classic dataset used for introducing machine learning concepts and techniques. This dataset contains information about the passengers aboard the Titanic, including features such as age, gender, ticket class, and whether or not they survived the disaster. The primary objective is to build a predictive model that can accurately classify whether a passenger survived or not based on these features.

    Machine learning offers a range of algorithms that can be applied to this classification problem. These algorithms can be broadly categorized into supervised learning techniques, where the model is trained on a labeled dataset. For the Titanic dataset, this means using the known outcomes (survived or not) to train the model.

    Key steps in applying machine learning to the Titanic dataset include:

    1. **Data Preprocessing:** This involves cleaning the data, handling missing values, and performing feature engineering to create relevant features that will improve the model's performance. The preprocessing steps ensure that the data is in a suitable format for training.
    2. **Splitting the Data:** The dataset is typically split into a training set and a test set. The training set is used to train the model, while the test set is used to evaluate its performance.
    3. **Selecting and Training Models:** Various machine learning algorithms can be applied to the Titanic dataset, including:
       - **Logistic Regression:** A simple and interpretable algorithm suitable for binary classification problems.
       - **Decision Trees:** A non-linear model that captures complex interactions between features.
       - **Random Forests:** An ensemble method that builds multiple decision trees and combines their predictions for improved accuracy.
       - **Support Vector Machines (SVM):** A powerful classifier that can find the optimal boundary between classes.
       - **Gradient Boosting:** An ensemble technique that builds models sequentially to correct errors made by previous models.

    4. **Model Evaluation:** The performance of the models is evaluated using metrics such as accuracy, precision, recall, and the F1 score. Cross-validation techniques can also be employed to ensure the model's robustness and to prevent overfitting.
    5. **Hyperparameter Tuning:** This involves optimizing the parameters of the chosen algorithms to improve their performance. Techniques like grid search or random search can be used for this purpose.
    6. **Making Predictions:** Once the model is trained and evaluated, it can be used to make predictions on new, unseen data. In the case of the Titanic dataset, this would involve predicting the survival of passengers based on their features.

    By applying machine learning techniques to the Titanic dataset, we can gain valuable insights into the factors that influenced survival and develop predictive models that can be used for similar classification tasks in other domains. The process also provides a practical introduction to key machine learning concepts and methods.
    """
    mle_models = """
    **Model Performance Evaluation Results**

    The table below presents the performance metrics for various machine learning models applied to the Titanic dataset. The metrics include Accuracy, Precision, Recall, F1-Score, AUC (Area Under the ROC Curve), and the Time taken for training and evaluation. Each metric provides insights into different aspects of model performance.


    **Explanation of Metrics:**

    - **Accuracy:** The proportion of correctly classified instances among the total instances. A higher value indicates better overall performance.
    - **Precision:** The proportion of true positive predictions among all positive predictions. It reflects the model's ability to avoid false positives.
    - **Recall:** The proportion of true positive predictions among all actual positives. It indicates the model's ability to capture all relevant instances (sensitivity).
    - **F1-Score:** The harmonic mean of precision and recall, providing a balance between the two. It is particularly useful when the class distribution is imbalanced.
    - **AUC (Area Under the ROC Curve):** Measures the model's ability to distinguish between classes. A higher AUC value indicates better performance.
    - **Time:** The time taken to train and evaluate the model.

    **Model Insights:**

    1. **Random Forest:** Achieved a high AUC of 0.886, indicating excellent discrimination between classes, with a good balance of precision and recall.
    2. **Logistic Regression:** Performed similarly to Random Forest with high accuracy and a strong F1-Score, but slightly lower AUC.
    3. **KNN:** Showed the highest accuracy and a strong F1-Score, but with a marginally lower AUC compared to Random Forest and Logistic Regression.
    4. **LGBM:** Performed well but with a slightly lower accuracy and AUC than Random Forest, Logistic Regression, and KNN.
    5. **AdaBoost:** Had decent performance but was slightly less effective in terms of precision and AUC compared to the top models.
    6. **Decision Tree:** Demonstrated good precision and recall but with a significantly lower AUC.
    7. **GaussianNB:** Had the lowest performance metrics, indicating poor model performance, especially with a very high recall but low precision and AUC.

    **Conclusion:**

    Among the models evaluated, the Random Forest, Logistic Regression, and KNN classifiers showed the best overall performance, with high accuracy, precision, recall, F1-Score, and AUC values. Random Forest had the highest AUC, making it the best model for distinguishing between classes. Logistic Regression and KNN also performed well, with KNN achieving the highest accuracy. The time metric indicates that Logistic Regression is the fastest to train and evaluate, followed by KNN, making them efficient choices for quick model training. GaussianNB showed the poorest performance, highlighting its unsuitability for this specific classification task.
    """

class ImagesURL:
    titanic = "https://raw.githubusercontent.com/fralfaro/posit-tables-2024/main/images/titanic.png"
class DataURL:
    titanic_train = "https://raw.githubusercontent.com/fralfaro/ploomber-example/main/data/train.csv"
    titanic_test = "https://raw.githubusercontent.com/fralfaro/ploomber-example/main/data/test.csv"

# extra functions

# tables

def calculate_percentage_vo_int(data, column, vo):
    """
    Calcula los porcentajes relativos de cada grupo dividido por una variable objetivo (vo) en un DataFrame,
    manteniendo la columna de interés como índice en la tabla pivote.

    Parámetros:
    data (pandas DataFrame): El DataFrame que contiene los datos.
    column (str): El nombre de la columna para la cual se calcularán los porcentajes.
    vo (str): El nombre de la variable objetivo para agrupar los datos y calcular los porcentajes.

    Returns:
    pandas DataFrame: Una tabla pivotante con los porcentajes relativos.
    """

    # Calcular el conteo de cada grupo y reestructurar los datos
    counts = data.groupby([vo, column]).size().reset_index(name='Count')

    # Calcular los porcentajes por categoría
    counts['Percentage'] = counts.groupby(vo)['Count'].transform(lambda x: (x / x.sum()))

    # Crear una tabla pivote con los porcentajes
    pivot_counts = counts.pivot_table(values=['Count', 'Percentage'], index=column, columns=vo)

    return pivot_counts

def calculate_percentage_vo(data, column, bins, vo):
    """
    Calcula los porcentajes relativos de cada grupo dividido por una variable objetivo (vo) en un DataFrame,
    dentro de rangos específicos definidos por column y bins.

    Parámetros:
    data (pandas DataFrame): El DataFrame que contiene los datos.
    column (str): El nombre de la columna para la cual se calcularán los porcentajes.
    bins (int or sequence of scalars): El número de contenedores (bins) o los límites de los contenedores para la división.
    vo (str): El nombre de la variable objetivo para agrupar los datos y calcular los porcentajes.

    Returns:
    pandas DataFrame: Una tabla pivotante con los porcentajes relativos.
    """

    # Agregar una nueva columna al DataFrame con los rangos de la columna específica
    data[column + 'Range'] = pd.cut(data[column], bins=bins, right=False)

    # Calcular el conteo de cada grupo y reestructurar los datos
    counts = data.groupby([vo, column + 'Range']).size().reset_index(name='Count')

    # Calcular los porcentajes por categoría
    counts['Percentage'] = counts.groupby(vo)['Count'].transform(lambda x: (x / x.sum()))

    # Crear una tabla pivote con los porcentajes
    pivot_counts = counts.pivot_table(values=['Count', 'Percentage'], index=column + 'Range', columns=vo)

    # eliminar columna extra
    data.drop(column + 'Range', axis=1, inplace=True)

    return pivot_counts

# plots
def plot_numeric_variables(df):
    st.markdown("""<big>Numeric Variables </big>""", unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    x_column_options = ["Age", "Fare"]  # You can add more columns if you wish
    x_column = col1.selectbox("Select column:", options=x_column_options)
    plot_type_options = [
        "univariate",
        "bivariate",
        "table"
    ]  # You can add more plot types if you wish
    plot_type = col2.selectbox("Numeric Plot type:", options=plot_type_options)

    if plot_type == "univariate":
        # Create a histogram plot with Plotly
        fig = px.histogram(df, x=x_column, title=f"Histogram of {x_column}")
        # Display the plot in Streamlit
        fig.update_traces(
            marker=dict(line=dict(color="black", width=1)), marker_color="skyblue"
        )
        st.plotly_chart(fig)
    elif plot_type == "bivariate":
        # Create a grouped histogram plot with Plotly
        fig = px.histogram(
            df,
            x=x_column,
            color="Survived",
            title=f"Histogram of {x_column} by Survived",
        )
        # Display the plot in Streamlit
        fig.update_traces(marker=dict(line=dict(color="black", width=1)))
        st.plotly_chart(fig)
    else:
        bins = {
            'Age':[0, 10, 20, 30, 40, 50, 60, 70, 80, 90],
            'Fare':[0, 10, 25, 50, 100, 1000]

        }
        hue = 'Survived'

        table = calculate_percentage_vo(df, x_column, bins[x_column], hue)

        # Reset the index
        # Flattening the MultiIndex columns
        table.columns = [f'{i}_{j}' for i, j in table.columns]

        # Resetting the index to make 'FareRange' a column again
        table.reset_index(inplace=True)

        # Displaying the DataFrame
        gt_table= ((
            GT(table)
            .tab_header(
                title="Count and Percentage Table",
                subtitle=f"{x_column} vs {hue}"
            )

            .tab_spanner(
                label="0",
                columns=["Count_0", "Percentage_0"]
            )
            .tab_spanner(
                label="1",
                columns=["Count_1", "Percentage_1"]
            )
            .tab_spanner(
                label=hue,
                columns=["Count_0", "Percentage_0", "Count_1", "Percentage_1"]
            )

            .cols_label(
                Count_0=html("Count"),
                Count_1=html("Count"),
                Percentage_0=html("Percentage"),
                Percentage_1=html("Percentage")

            )
        )
        )
        gt_table = gt_table.fmt_number(columns=["Percentage_0", "Percentage_1"],
                                       decimals=2)  # .opt_stylize(style = 1, color = "blue")



        gt_table = gt_table.tab_options(
            table_background_color="white",
            #table_font_color="darkblue",
            table_font_style="italic",
            table_font_names="Times New Roman",
            heading_background_color="skyblue"
        )

        st.html(gt_table.as_raw_html())


def plot_categorical_variables(df):
    st.markdown("""<big>Categorical Variables </big>""", unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    x_column_options_cat = [
        "Pclass",
        "SibSp",
        "Parch",
        "Sex",
        "Cabin",
        "Embarked",
    ]  # You can add more columns if you wish
    x_column_cat = col1.selectbox("Select column:", options=x_column_options_cat)
    plot_type_options_cat = [
        "univariate",
        "bivariate",
        "table",
    ]  # You can add more plot types if you wish
    plot_type_cat = col2.selectbox("Categorical Plot type:", options=plot_type_options_cat)

    if plot_type_cat == "univariate":
        # Create a bar plot with Altair
        chart = alt.Chart(df).mark_bar(
            color='skyblue',
            stroke='black',
            strokeWidth=1
        ).encode(
            alt.X(x_column_cat, title=f"{x_column_cat}"),
            y=alt.Y('count()', title='Count'),
            tooltip=[x_column_cat, 'count()']
        ).properties(
            title=f"Barplot of {x_column_cat}",
            width=600,
            height=400
        ).configure_title(
            fontSize=20,
            anchor='start',
            color='gray'
        ).configure_axis(
            labelFontSize=12,
            titleFontSize=14
        ).configure_axisX(
            labelAngle=0
        ).configure_view(
            strokeWidth=0
        )
        # Display the plot in Streamlit
        st.altair_chart(chart, use_container_width=True)
    elif plot_type_cat == "bivariate":
        # Create a grouped bar plot with Altair
        chart = alt.Chart(df).mark_bar(
            stroke='black',
            strokeWidth=1
        ).encode(
            alt.X(x_column_cat, title=f"{x_column_cat}"),
            y=alt.Y('count()', title='Count'),
            color=alt.Color('Survived:N', legend=alt.Legend(title='Survived')),
            tooltip=[x_column_cat, 'count()', 'Survived']
        ).properties(
            title=f"Barplot of {x_column_cat} by Survived",
            width=600,
            height=400
        ).configure_title(
            fontSize=20,
            anchor='start',
            color='gray'
        ).configure_axis(
            labelFontSize=12,
            titleFontSize=14
        ).configure_axisX(
            labelAngle=0
        ).configure_view(
            strokeWidth=0
        )
        # Display the plot in Streamlit
        st.altair_chart(chart, use_container_width=True)
    else:

        hue = 'Survived'

        table = calculate_percentage_vo_int(df,x_column_cat,hue).fillna(0)

        # Reset the index
        # Flattening the MultiIndex columns
        table.columns = [f'{i}_{j}' for i, j in table.columns]

        # Resetting the index to make 'FareRange' a column again
        table.reset_index(inplace=True)

        # Displaying the DataFrame
        gt_table= ((
            GT(table)
            .tab_header(
                title="Count and Percentage Table",
                subtitle=f"{x_column_cat} vs {hue}"
            )

            .tab_spanner(
                label="0",
                columns=["Count_0", "Percentage_0"]
            )
            .tab_spanner(
                label="1",
                columns=["Count_1", "Percentage_1"]
            )
            .tab_spanner(
                label=hue,
                columns=["Count_0", "Percentage_0", "Count_1", "Percentage_1"]
            )

            .cols_label(
                Count_0=html("Count"),
                Count_1=html("Count"),
                Percentage_0=html("Percentage"),
                Percentage_1=html("Percentage")

            )
        )
        )
        gt_table = gt_table.fmt_number(columns=["Percentage_0", "Percentage_1"],
                                       decimals=2)  # .opt_stylize(style = 1, color = "blue")

        gt_table = gt_table.tab_options(
            table_background_color="white",
            # table_font_color="darkblue",
            table_font_style="italic",
            table_font_names="Times New Roman",
            heading_background_color="skyblue"
        )

        st.html(gt_table.as_raw_html())


def preprocess_titanic_data(df):
    """
    Preprocess the Titanic dataset.

    Parameters:
    df (pd.DataFrame): The input dataframe.

    Returns:
    pd.DataFrame: The preprocessed dataframe.
    """
    # Drop columns
    cols_delete = ['Name', 'Ticket']
    df = df.drop(cols_delete, axis=1)

    # Fill missing values in 'Age' with the mean
    age_mean = round(df['Age'].mean())
    df['Age'] = df['Age'].fillna(age_mean)

    # Fill missing values in 'Cabin' with 'N' and keep only the first letter
    df['Cabin'] = df['Cabin'].fillna('N').str[0]

    # Convert specified columns to string type
    columns_to_convert = ['Pclass', 'SibSp', 'Parch']
    df[columns_to_convert] = df[columns_to_convert].astype(str)

    return df

# Define img_to_bytes() function
def img_to_bytes(img_url):
    response = requests.get(img_url)
    img_bytes = response.content
    encoded = base64.b64encode(img_bytes).decode()
    return encoded



# main function
def main():
    """
    Main function to set up the Streamlit app layout.
    """
    cs_sidebar()
    cs_body()
    return None



def plot_confusion_matrix(y_true, y_pred, class_names):
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Create confusion matrix plot using Plotly
    fig = ff.create_annotated_heatmap(
        z=cm, x=class_names, y=class_names, colorscale="Blues", showscale=False
    )

    # Add labels
    fig.update_layout(
        title="Confusion Matrix",
        xaxis=dict(title="Predicted Value"),
        yaxis=dict(title="True Value"),
        width=400,
        height=400,
    )

    return fig


def plot_roc_curve(y_true, y_score):
    # Calculate the ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    # Create a Plotly figure for the ROC curve
    fig = go.Figure()

    # Add the ROC curve with customized color
    fig.add_trace(
        go.Scatter(
            x=fpr,
            y=tpr,
            mode="lines",
            name=f"ROC Curve (AUC = {roc_auc:.2f})",
            line=dict(color="skyblue"),
        )
    )

    # Add diagonal line for reference
    fig.add_shape(
        type="line",
        x0=0,
        y0=0,
        x1=1,
        y1=1,
        line=dict(color="lightblue", dash="dash"),
        name="Reference (AUC = 0.5)",
    )

    # Customize the layout of the figure
    fig.update_layout(
        title="ROC Curve",
        xaxis=dict(title="False Positive Rate"),
        yaxis=dict(title="True Positive Rate"),
        width=400,
        height=400,
    )
    return fig


# Define the cs_sidebar() function
def cs_sidebar():
    """
    Populate the sidebar with various content sections related to Python.
    """

    st.sidebar.markdown(
        """[<img src='data:image/png;base64,{}' class='img-fluid' width=200 >](https://streamlit.io/)""".format(
            img_to_bytes(
                ImagesURL.titanic
            )
        ),
        unsafe_allow_html=True,
    )

    st.sidebar.header("Titanic - ML from Disaster")
    st.sidebar.markdown("""
    [![documentation](https://img.shields.io/badge/📖-docs-brightgreen)](https://fralfaro.github.io/Titanic-Challenge/)
    [![documentation](https://img.shields.io/badge/🌎-blog-blue)](https://fralfaro.github.io/Titanic-Challenge/)
    """)

    st.sidebar.markdown(SidebarText.introduction,unsafe_allow_html=True)

    st.sidebar.markdown("__🛳️Goals__")
    st.sidebar.markdown(SidebarText.introduction,unsafe_allow_html=True)

    return None


# Define the cs_body() function
def cs_body():
    """
    Create content sections for the main body of the
     Streamlit cheat sheet with Python examples.
    """

    @st.cache_data()
    def load_data():
        # Load data from CSV file
        data = pd.read_csv(DataURL.titanic_train)
        # Convert certain columns to string type
        columns_to_convert = ["Pclass", "SibSp", "Parch"]
        data[columns_to_convert] = data[columns_to_convert].astype(str)
        # Fill missing values in 'Cabin' column with '-' and extract the first character
        data["Cabin"] = data["Cabin"].fillna("-").apply(lambda x: x[0])
        # Convert certain columns to string type again
        columns_to_convert = ["Pclass", "SibSp", "Parch"]
        data[columns_to_convert] = data[columns_to_convert].astype(str)
        return data

    # Title of the application
    st.title("Titanic EDA with Streamlit and Plotly")

    # Tab menu.
    tab1, tab2, tab3 = st.tabs(
        ["📊 Exploratory Data Analysis", "📝 Feature Engineering", "🤖 Machine Learning"]
    )

    # Load Data: train.csv
    df = load_data()



    with tab1:
        st.markdown(BodyText.eda_intro, unsafe_allow_html=True)

        st.subheader("Data")
        interactive_table(
            df,
            buttons=["copyHtml5", "csvHtml5", "excelHtml5"],
            maxBytes = 0
        )


        st.subheader("Plots")

        # numerical variables
        plot_numeric_variables(df)

        # categorical variables
        plot_categorical_variables(df)


    with tab2:
        st.markdown(BodyText.fe_intro, unsafe_allow_html=True)

        # Crear dos columnas
        col1, col2 = st.columns(2)

        new_df = preprocess_titanic_data(df)

        # Botón en la primera columna
        with col1:
            st.subheader('Before')
            st.write(df)

        # Botón en la segunda columna
        with col2:
            st.subheader('After')
            st.write(new_df)

        st.markdown(BodyText.fe_conclusion, unsafe_allow_html=True)

    with tab3:

        st.markdown(BodyText.mle_intro, unsafe_allow_html=True)

    css = '''
    <style>
        .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size:1.5rem;
        }
    </style>
    '''

    st.markdown(css, unsafe_allow_html=True)





# Run the main function if the script is executed directly
if __name__ == "__main__":
    main()