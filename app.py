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
        <small>The [RMS Titanic](https://es.wikipedia.org/wiki/RMS_Titanic)
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
    pass
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

    # Load Data: train.csv
    df = load_data()

    #col1, col2, col3, col4 = st.columns(4)
    interactive_table(
        df,
        buttons=["copyHtml5", "csvHtml5", "excelHtml5"],
        maxBytes = 0
    )


    # Create a selectbox widget to choose between 'head' and 'tail'
    #option = col1.selectbox("Select 'head' or 'tail'", options=["head", "tail"])
    # Create a number_input widget to select the number of rows
    #num_rows = col2.slider(
    #    "Number of rows to display", min_value=1, max_value=50, value=5
    #)
    # Function to display the DataFrame according to the selected option
    #def display_dataframe(option, num_rows):
    #    if option == "head":
    #        st.write(df.head(num_rows))
    #    elif option == "tail":
    #        st.write(df.tail(num_rows))



    # Display the DataFrame based on the selected option
    #display_dataframe(option, num_rows)

    st.subheader("Plots")

    # numerical variables
    plot_numeric_variables(df)

    # categorical variables
    plot_categorical_variables(df)

    st.subheader("Models and Metrics")






# Run the main function if the script is executed directly
if __name__ == "__main__":
    main()