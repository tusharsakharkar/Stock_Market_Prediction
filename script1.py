from flask import Flask, render_template, request
from bokeh.sampledata.glucose import data
import pandas as pd
app=Flask(__name__)

@app.route('/plot/',methods =['POST'])
def plot():
    from pandas_datareader import data
    import datetime
    from bokeh.plotting import figure, show, output_file,vplot
    from bokeh.embed import components
    from bokeh.resources import CDN
    from bokeh.models import HoverTool

    #  output_file('plot.html')
    start = datetime.datetime(2014,12,30)

    end = datetime.datetime.today().strftime("%Y/%m/%d")
    print(end)
    word = request.form['company_ip']

    df=data.DataReader(name=word,data_source='morningstar',start=start,end=end)



    def inc_dec(c, o):
        if c > o:
            value="Increase"
        elif c < o:
            value="Decrease"
        else:
            value="Equal"
        return value

    df["Status"]=[inc_dec(c,o) for c, o in zip(df.Close,df.Open)]
    df["Middle"]=(df.Open+df.Close)/2
    df["Height"]=abs(df.Close-df.Open)
    p=figure( x_axis_type="datetime", width=1200, height=500)
    p.title='Candlestick Chart'
    hours_12 = 12 * 60 * 60 * 1000
#    print(df.index.levels[df.Status=="Increase"])
    #    p.line([1, 2, 3, 4, 5], [6, 7, 2, 4, 5], line_width=2)
#    pp = figure(width=1000, height=300, responsive=True)
#    p.multi_line([[1, 3, 2], [3, 4, 6, 6]], [[2, 1, 4], [4, 7, 8, 5]],color=["firebrick", "navy"], alpha=[0.8, 0.3], line_width=4)

#   p.segment(df.index, df.High, df.index, df.Low, color="Black")


    #p.line(df.Low, df.High, color='navy', alpha=0.5)

#    p1= figure(x_axis_type="datetime", width=1000, height=300, responsive=True)
#    p1.title = 'Candlestick Chart'
#   p1.rect(df.High[df.Status=="Increase"],df.Middle[df.Status=="Increase"],hours_12, df.Height[df.Status=="Increase"],fill_color="#CCFFFF",line_color="black")
#    print(df.Height[df.Status=="Increase"])
#  show(p)


#    p.rect(df.index[df.Status=="Increase"],df.Middle[df.Status=="Increase"],hours_12, df.Height[df.Status=="Increase"],fill_color="#FF3333",line_color="black")
    print(df)
    months = ["Jan16", "June16", "Jan17", "June17", "Jan18"]
    p.segment(df.index, df.High, df.index, df.Low, color="Black")
    p.rect(df.index.levels[1], df.Middle[df.Status == "Increase"],hours_12, df.Height[df.Status == "Increase"], fill_color="#CCFFFF", line_color="black")

    plot = figure(plot_width=1000, plot_height=500, x_axis_type="datetime", tools="", toolbar_location=None,
                  title='Hover over points')
    plot.line(df.index.levels[1], df.Close, line_dash="4 4", line_width=3, color='Black',alpha=5.5)
    cr = plot.circle(df.index.levels[1], df.Close, size=20,
                     fill_color="grey", hover_fill_color="firebrick",
                     fill_alpha=0.05, hover_alpha=0.3,
                     line_color=None, hover_line_color="white")
    plot.add_tools(HoverTool(tooltips=None, renderers=[cr], mode='hline'))

    s1 = figure(x_axis_type="datetime",width=1000, plot_height=500, title=None)
    s1.line(df.index.levels[1], df.Middle[df.Status=="Increase"], color='navy', alpha=0.5)
    plot.title = "Rise in stock"
    plot.grid.grid_line_alpha = 0
    plot.xaxis.axis_label = 'Date'
    plot.yaxis.axis_label = 'Price'
    plot.ygrid.band_fill_color = "olive"
    plot.ygrid.band_fill_alpha = 0.1

    # create another one
    s2 = figure(x_axis_type="datetime",width=1000, plot_height=500, title=None)
#   s2.line(df.index.levels[1], df.Middle[df.Status=="Decrease"], color='navy', alpha=0.5)
    s2.circle(df.index.levels[1], df.Middle[df.Status=="Decrease"], size=10, color="navy", alpha=0.5)
    s2.title = "Fall in stock"
    s2.grid.grid_line_alpha = 0
    s2.xaxis.axis_label = 'Date'
    s2.yaxis.axis_label = 'Price'
    s2.ygrid.band_fill_color = "olive"
    s2.ygrid.band_fill_alpha = 0.1

    # create and another
    s3 = figure(x_axis_type="datetime",width=1000, plot_height=500, title=None)
    s3.line(df.index.levels[1], df.Close, color='navy', alpha=0.5)
    s3.title = "Stock Closing Prices"
    s3.grid.grid_line_alpha = 0
    s3.xaxis.axis_label = 'Date'
    s3.yaxis.axis_label = 'Price'
    s3.ygrid.band_fill_color = "olive"
    s3.ygrid.band_fill_alpha = 0.1



    vv = vplot(plot,s2,s3)
    script1, div1 = components(vv)
    cdn_js=CDN.js_files[0]
    cdn_css=CDN.css_files[0]
    predicted = predict_prices(df.Middle[-3:],29)
    print(predicted)
    return render_template("plot.html",
    script1=script1,
    predicted = predicted,
    div1=div1,
    cdn_css=cdn_css,
    cdn_js=cdn_js)
    

    
def predict_prices(prices,x):
    from sklearn.svm import SVR
    import numpy as np
    
    dates = list(range(len(prices)))
    dates = np.reshape(dates,(len(dates),1))
    svr_rbf = SVR(kernel = 'rbf',C=1e3,gamma=0.2)
    svr_rbf.fit(dates,prices)

    return svr_rbf.predict(x)[0]


@app.route('/')
def home():
    return render_template('index.html')


if __name__=="__main__":
    app.run(debug=True)
