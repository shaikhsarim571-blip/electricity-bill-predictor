import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from config import COLOR_PALETTE, THEME_PRIMARY, THEME_ACCENT_BLUE


class Visualizations:
    def __init__(self, config):
        self.config = config
        self.color_map = COLOR_PALETTE
    
    def correlation_heatmap(self, df, numeric_cols=None):
        """Interactive correlation heatmap with Plotly"""
        if numeric_cols is None:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        corr_matrix = df[numeric_cols].corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            zmin=-1,
            zmax=1,
            text=corr_matrix.values.round(2),
            texttemplate='%{text}',
            textfont={"size": 10, "color": "white"},
            hovertemplate='%{y} vs %{x}<br>Correlation: %{z:.3f}<extra></extra>',
            colorbar=dict(title="Pearson<br>Correlation")
        ))
        
        fig.update_layout(
            title="Feature Correlation Matrix",
            xaxis_title="Features",
            yaxis_title="Features",
            height=600,
            template="plotly_dark",
            paper_bgcolor=THEME_PRIMARY,
            plot_bgcolor=THEME_PRIMARY,
            font=dict(color="white", size=11),
            xaxis=dict(tickfont=dict(color="white")),
            yaxis=dict(tickfont=dict(color="white"))
        )
        
        return fig
    
    def seasonal_decomposition_plot(self, decomposition, title="Consumption Decomposition"):
        """Seasonal decomposition subplots"""
        
        fig = make_subplots(
            rows=4, cols=1,
            subplot_titles=("Original", "Trend", "Seasonal", "Residual"),
            shared_xaxes=True,
            vertical_spacing=0.08
        )
        
        fig.add_trace(
            go.Scatter(x=decomposition.observed.index, y=decomposition.observed.values,
                      name="Original", line=dict(color=THEME_ACCENT_BLUE, width=2), mode='lines'),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=decomposition.trend.index, y=decomposition.trend.values,
                      name="Trend", line=dict(color="#FFD700", width=2), mode='lines'),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=decomposition.seasonal.index, y=decomposition.seasonal.values,
                      name="Seasonal", line=dict(color="#39FF14", width=2), mode='lines'),
            row=3, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=decomposition.resid.index, y=decomposition.resid.values,
                      name="Residual", line=dict(color="#FF4444", width=2), mode='lines'),
            row=4, col=1
        )
        
        fig.update_yaxes(title_text="kWh", row=1, col=1, tickfont=dict(color="white"), gridcolor="#333333")
        fig.update_yaxes(title_text="Trend", row=2, col=1, tickfont=dict(color="white"), gridcolor="#333333")
        fig.update_yaxes(title_text="Seasonal", row=3, col=1, tickfont=dict(color="white"), gridcolor="#333333")
        fig.update_yaxes(title_text="Residual", row=4, col=1, tickfont=dict(color="white"), gridcolor="#333333")
        
        fig.update_xaxes(tickfont=dict(color="white"), gridcolor="#333333")
        
        fig.update_layout(
            title=dict(text=title, font=dict(color="white", size=16)),
            height=900,
            template="plotly_dark",
            paper_bgcolor=THEME_PRIMARY,
            plot_bgcolor=THEME_PRIMARY,
            showlegend=False,
            font=dict(color="white", size=11),
            hovermode='x unified',
            yaxis=dict(tickfont=dict(color="white"))
        )
        
        return fig
    
    def model_comparison_chart(self, comparison_df):
        """Bar chart comparing model metrics"""
        metrics = comparison_df.columns.tolist()
        
        fig = make_subplots(
            rows=1, cols=len(metrics),
            subplot_titles=metrics
        )
        
        colors = ["#00D9FF", "#39FF14"]
        
        for idx, metric in enumerate(metrics, 1):
            for i, model in enumerate(comparison_df.index):
                fig.add_trace(
                    go.Bar(
                        x=[model],
                        y=[comparison_df.loc[model, metric]],
                        name=model,
                        marker=dict(color=colors[i]),
                        showlegend=(idx == 1)
                    ),
                    row=1, col=idx
                )
            
            fig.update_xaxes(title_text="Model", row=1, col=idx)
            fig.update_yaxes(title_text=metric, row=1, col=idx)
        
        fig.update_layout(
            title="Model Performance Comparison",
            height=400,
            template="plotly_dark",
            paper_bgcolor=THEME_PRIMARY,
            plot_bgcolor=THEME_PRIMARY,
            font=dict(color="white"),
            xaxis=dict(tickfont=dict(color="white")),
            yaxis=dict(tickfont=dict(color="white")),
            barmode='group'
        )
        
        return fig
    
    def actual_vs_predicted(self, y_test, y_pred, forecast_future=None):
        """Actual vs Predicted with optional future forecast"""
        
        fig = go.Figure()
        
        x_actual = list(range(len(y_test)))
        fig.add_trace(go.Scatter(
            x=x_actual,
            y=y_test.values if hasattr(y_test, 'values') else y_test,
            name='Actual',
            line=dict(color="#39FF14", width=2),
            mode='lines+markers'
        ))
        
        fig.add_trace(go.Scatter(
            x=x_actual,
            y=y_pred,
            name='Predicted',
            line=dict(color="#00D9FF", width=2),
            mode='lines+markers'
        ))
        
        if forecast_future is not None and len(forecast_future) > 0:
            x_future = list(range(len(y_test), len(y_test) + len(forecast_future)))
            fig.add_trace(go.Scatter(
                x=x_future,
                y=forecast_future,
                name='3-Month Forecast',
                line=dict(color="#00D9FF", width=2, dash='dash'),
                fill='tozeroy',
                fillcolor='rgba(0, 217, 255, 0.2)'
            ))
            
            fig.add_vrect(
                x0=len(y_test)-0.5, x1=len(y_test)+len(forecast_future)-0.5,
                fillcolor="#00D9FF", opacity=0.1,
                annotation_text="Forecast Window", annotation_position="top left"
            )
        
        fig.update_layout(
            title="Actual vs Predicted Consumption",
            xaxis_title="Time Period",
            yaxis_title="Consumption (kWh)",
            height=500,
            template="plotly_dark",
            paper_bgcolor=THEME_PRIMARY,
            plot_bgcolor=THEME_PRIMARY,
            font=dict(color="white"),
            xaxis=dict(tickfont=dict(color="white")),
            yaxis=dict(tickfont=dict(color="white")),
            hovermode='x unified'
        )
        
        return fig
    
    def cluster_scatter_2d(self, pca_data, labels, feature_names):
        """2D scatter plot of clusters from PCA"""
        
        df_viz = pd.DataFrame({
            'PC1': pca_data[:, 0],
            'PC2': pca_data[:, 1],
            'Cluster': labels
        })
        
        fig = px.scatter(
            df_viz,
            x='PC1',
            y='PC2',
            color='Cluster',
            color_discrete_sequence=[
                self.color_map.get(f'cluster_{i}', f'#00D9FF')
                for i in range(len(np.unique(labels)))
            ],
            title="Consumer Segmentation (PCA Projection)",
            labels={'PC1': f'PC1 ({feature_names[0]})', 'PC2': f'PC2 ({feature_names[1]})'},
            height=600
        )
        
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor=THEME_PRIMARY,
            plot_bgcolor=THEME_PRIMARY,
            font=dict(color="white", size=12),
            xaxis=dict(tickfont=dict(color="white")),
            yaxis=dict(tickfont=dict(color="white")),
            hovermode='closest'
        )
        
        fig.update_traces(marker=dict(size=10, opacity=0.7))
        
        return fig
    
    def cluster_scatter_3d(self, pca_data, labels):
        """3D scatter plot if PCA has 3 components"""
        if pca_data.shape[1] < 3:
            return None
        
        df_viz = pd.DataFrame({
            'PC1': pca_data[:, 0],
            'PC2': pca_data[:, 1],
            'PC3': pca_data[:, 2],
            'Cluster': labels
        })
        
        fig = px.scatter_3d(
            df_viz,
            x='PC1',
            y='PC2',
            z='PC3',
            color='Cluster',
            color_discrete_sequence=[
                self.color_map.get(f'cluster_{i}', f'#00D9FF')
                for i in range(len(np.unique(labels)))
            ],
            title="Consumer Segmentation (3D PCA)",
            height=600
        )
        
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor=THEME_PRIMARY,
            plot_bgcolor=THEME_PRIMARY,
            font=dict(color="white", size=12),
            xaxis=dict(tickfont=dict(color="white")),
            yaxis=dict(tickfont=dict(color="white"))
        )
        
        return fig
    
    def bill_breakdown(self, bill_data):
        """Pie chart for bill breakdown"""
        
        fig = go.Figure(data=[go.Pie(
            labels=bill_data['Component'],
            values=bill_data['Amount (PKR)'],
            marker=dict(colors=['#00D9FF', '#39FF14', '#FFD700', '#FF4444']),
            textposition='inside',
            textinfo='label+percent'
        )])
        
        fig.update_layout(
            title="Bill Breakdown",
            height=500,
            template="plotly_dark",
            paper_bgcolor=THEME_PRIMARY,
            plot_bgcolor=THEME_PRIMARY,
            font=dict(color="white"),
            showlegend=True
        )
        
        fig.update_traces(textfont=dict(color="white"))
        
        return fig
    
    def consumption_by_season(self, df):
        """Box plot of consumption by season"""
        
        df_seasonal = df.copy()
        
        if 'season' not in df_seasonal.columns:
            df_seasonal['date'] = pd.to_datetime(df_seasonal.get('date', pd.date_range('2020-01-01', periods=len(df))))
            season_map = {12: 'Winter', 1: 'Winter', 2: 'Winter',
                         3: 'Spring', 4: 'Spring', 5: 'Spring',
                         6: 'Summer', 7: 'Summer', 8: 'Summer',
                         9: 'Fall', 10: 'Fall', 11: 'Fall'}
            df_seasonal['season'] = df_seasonal['date'].dt.month.map(season_map)
        
        fig = px.box(
            df_seasonal,
            x='season',
            y='unit_consumption_kwh',
            title="Consumption Distribution by Season",
            category_orders={'season': ['Winter', 'Spring', 'Summer', 'Fall']}
        )
        
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor=THEME_PRIMARY,
            plot_bgcolor=THEME_PRIMARY,
            font=dict(color="white"),
            xaxis=dict(tickfont=dict(color="white")),
            yaxis=dict(tickfont=dict(color="white")),
            xaxis_title="Season",
            yaxis_title="Consumption (kWh)"
        )
        
        return fig
