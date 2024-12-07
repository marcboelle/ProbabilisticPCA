import plotly.graph_objects as go

def plot_fig_projections(X, projections_2d, line_scale=1):

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=X[:, 0],
        y=X[:, 1],
        mode='markers',
        marker=dict(size=12, color='blue', line=dict(width=1, color='black')),
        name="Original data",
        showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=projections_2d[:, 0],
        y=projections_2d[:, 1],
        mode='markers',
        marker=dict(size=8, color='darkorange'),
        name="Projections",
        showlegend=False
    ))

    min_projection = projections_2d.min(axis=0)
    max_projection = projections_2d.max(axis=0)

    slope = (max_projection[1] - min_projection[1])/(max_projection[0] - min_projection[0])
    fig.add_trace(go.Scatter(
        x=[min_projection[0]-line_scale, max_projection[0]+line_scale],
        y=[min_projection[1]-line_scale*slope, max_projection[1]+line_scale*slope],
        mode='lines',
        line=dict(color='darkorange', width=2),
        name="Principal direction",
        showlegend=False
    ))

    for point, projection in zip(X, projections_2d):
        fig.add_trace(go.Scatter(
            x=[point[0], projection[0]],
            y=[point[1], projection[1]],
            mode='lines',
            line=dict(color='gray', width=1),
            showlegend=False
        ))

    fig.update_layout(
        xaxis_title="X",
        yaxis_title="Y",
        template="plotly_white",
        plot_bgcolor="white",
        xaxis=dict(scaleanchor="y")
    )
    #fig.update_layout(height=600, width=1000)
    fig.update_layout(
        autosize=False,
        width=800,
        height=800,
    )
    fig.show()