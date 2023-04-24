from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField

class FlowerForm(FlaskForm):
    sepal_length = StringField('Sepal Length')
    sepal_width = StringField('Sepal Width')
    petal_length = StringField('Petal Length')
    petal_width = StringField('Petal Width')
    
    submit = SubmitField('Analyze')