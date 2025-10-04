import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.gridspec as gridspec
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import warnings
warnings.filterwarnings('ignore')

# Set up stylish theme
plt.style.use('dark_background')
sns.set_palette("viridis")

class MiningDetectionVisualizer:
    def __init__(self):
        self.fig = None
        self.colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8', '#F7DC6F']

    def generate_simulated_data(self):
        """Generate realistic simulated data for mining detection"""
        np.random.seed(42)

        # Ground truth and predictions
        n_samples = 500
        y_true = np.concatenate([
            np.ones(200),  # Mining areas
            np.zeros(300)  # Non-mining areas
        ])

        # Simulate predictions with some errors
        y_pred = y_true.copy()
        # Introduce some misclassifications
        misclass_indices = np.random.choice(np.where(y_true == 1)[0], 25, replace=False)
        y_pred[misclass_indices] = 0
        misclass_indices = np.random.choice(np.where(y_true == 0)[0], 35, replace=False)
        y_pred[misclass_indices] = 1

        return y_true, y_pred

    def create_advanced_confusion_matrix(self, y_true, y_pred):
        """Create a stunning confusion matrix visualization"""
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

        # Main confusion matrix heatmap
        im = ax1.imshow(cm, cmap='YlOrRd', interpolation='nearest', alpha=0.8)
        ax1.set_xticks([0, 1])
        ax1.set_yticks([0, 1])
        ax1.set_xticklabels(['Non-Mining', 'Mining'], fontsize=12, fontweight='bold')
        ax1.set_yticklabels(['Non-Mining', 'Mining'], fontsize=12, fontweight='bold')
        ax1.set_xlabel('Predicted Label', fontsize=14, fontweight='bold', color='white')
        ax1.set_ylabel('True Label', fontsize=14, fontweight='bold', color='white')
        ax1.set_title('Confusion Matrix - Mining Detection', fontsize=16, fontweight='bold', pad=20)

        # Add text annotations
        for i in range(2):
            for j in range(2):
                ax1.text(j, i, f'{cm[i, j]}',
                        ha="center", va="center",
                        color="white" if cm[i, j] < cm.max()/2 else "black",
                        fontsize=16, fontweight='bold')

        # Metrics breakdown
        metrics = ['True Negatives', 'False Positives', 'False Negatives', 'True Positives']
        values = [tn, fp, fn, tp]
        colors = ['#2E8B57', '#FF6347', '#FFD700', '#1E90FF']

        bars = ax2.barh(metrics, values, color=colors, alpha=0.8)
        ax2.set_xlabel('Count', fontsize=14, fontweight='bold', color='white')
        ax2.set_title('Classification Breakdown', fontsize=16, fontweight='bold', pad=20)
        ax2.grid(axis='x', alpha=0.3)

        # Add value labels on bars
        for bar, value in zip(bars, values):
            ax2.text(bar.get_width() + 5, bar.get_y() + bar.get_height()/2,
                    f'{value}', va='center', ha='left', fontsize=12, fontweight='bold', color='white')

        plt.tight_layout()
        return fig, cm

    def create_performance_radar(self, metrics_dict):
        """Create a radar chart for performance metrics"""
        categories = list(metrics_dict.keys())
        values = list(metrics_dict.values())

        # Complete the circle
        values += values[:1]
        categories += [categories[0]]

        angles = np.linspace(0, 2*np.pi, len(categories), endpoint=True).tolist() # Changed endpoint to True and removed appending the first angle

        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

        ax.plot(angles, values, 'o-', linewidth=2, color='#FF6B6B', label='Performance')
        ax.fill(angles, values, alpha=0.25, color='#FF6B6B')

        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_thetagrids(np.degrees(angles[:-1]), categories[:-1]) # Adjusted thetagrids to match categories length

        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.grid(True)
        ax.set_title('Model Performance Radar Chart', size=16, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))

        return fig

    def create_mining_heatmap_3d(self):
        """Create an interactive 3D heatmap of mining activity"""
        # Simulate mining intensity data
        x = np.linspace(0, 10, 20)
        y = np.linspace(0, 10, 20)
        X, Y = np.meshgrid(x, y)

        # Create mining hotspots
        Z = (np.exp(-((X-3)*2 + (Y-3)*2)) +
             np.exp(-((X-7)*2 + (Y-7)*2)/0.5) +
             np.exp(-((X-2)*2 + (Y-8)*2)/0.3))

        fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y, colorscale='Hot')])

        fig.update_layout(
            title='3D Mining Activity Heat Map',
            scene=dict(
                xaxis_title='Longitude',
                yaxis_title='Latitude',
                zaxis_title='Mining Intensity',
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
            ),
            width=800,
            height=600
        )

        return fig

    def create_comprehensive_dashboard(self, y_true, y_pred):
        """Create a comprehensive dashboard with all visualizations"""
        # Calculate metrics
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()

        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1_score = 2 * (precision * recall) / (precision + recall)
        iou = tp / (tp + fp + fn)

        # Simulate ROC curve
        y_scores = np.random.rand(len(y_true)) * 0.3 + y_pred * 0.7
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)

        # Create the main figure
        fig = plt.figure(figsize=(25, 20))
        gs = gridspec.GridSpec(3, 3, figure=fig)

        # 1. Confusion Matrix
        ax1 = fig.add_subplot(gs[0, 0])
        im = ax1.imshow(cm, cmap='RdYlBu_r', alpha=0.8)
        ax1.set_xticks([0, 1])
        ax1.set_yticks([0, 1])
        ax1.set_xticklabels(['Non-Mining', 'Mining'], fontsize=11, fontweight='bold')
        ax1.set_yticklabels(['Non-Mining', 'Mining'], fontsize=11, fontweight='bold')
        ax1.set_title('Confusion Matrix', fontsize=14, fontweight='bold', pad=15)
        for i in range(2):
            for j in range(2):
                ax1.text(j, i, f'{cm[i, j]}', ha="center", va="center",
                        color="white", fontsize=16, fontweight='bold')

        # 2. ROC Curve
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(fpr, tpr, color='#FF6B6B', lw=3, label=f'ROC (AUC = {roc_auc:.3f})')
        ax2.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', alpha=0.5)
        ax2.fill_between(fpr, tpr, alpha=0.3, color='#FF6B6B')
        ax2.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
        ax2.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
        ax2.set_title('ROC Curve', fontsize=14, fontweight='bold', pad=15)
        ax2.legend(loc='lower right')
        ax2.grid(True, alpha=0.3)

        # 3. Metrics Bar Chart
        ax3 = fig.add_subplot(gs[0, 2])
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'IoU']
        scores = [accuracy, precision, recall, f1_score, iou]
        bars = ax3.bar(metrics, scores, color=self.colors[:5], alpha=0.8)
        ax3.set_ylim(0, 1)
        ax3.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax3.set_title('Performance Metrics', fontsize=14, fontweight='bold', pad=15)
        ax3.grid(axis='y', alpha=0.3)
        for bar, score in zip(bars, scores):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{score:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=11)

        # 4. Mining Distribution Pie Chart
        ax4 = fig.add_subplot(gs[1, 0])
        labels = ['Legal Mining', 'Illegal Mining', 'Non-Mining Area']
        sizes = [45, 15, 40]
        colors = ['#2E8B57', '#FF6347', '#4682B4']
        wedges, texts, autotexts = ax4.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                                          startangle=90, textprops={'fontsize': 11, 'fontweight': 'bold'})
        ax4.set_title('Land Use Distribution', fontsize=14, fontweight='bold', pad=15)

        # 5. Mining Intensity Over Time
        ax5 = fig.add_subplot(gs[1, 1])
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct']
        legal_mining = [30, 32, 35, 38, 42, 45, 43, 46, 45, 45]
        illegal_mining = [8, 9, 10, 12, 13, 14, 15, 15, 15, 15]
        ax5.plot(months, legal_mining, marker='o', linewidth=3, label='Legal Mining', color='#2E8B57')
        ax5.plot(months, illegal_mining, marker='s', linewidth=3, label='Illegal Mining', color='#FF6347')
        ax5.fill_between(months, legal_mining, alpha=0.3, color='#2E8B57')
        ax5.fill_between(months, illegal_mining, alpha=0.3, color='#FF6347')
        ax5.set_xlabel('Month', fontsize=12, fontweight='bold')
        ax5.set_ylabel('Mining Area (hectares)', fontsize=12, fontweight='bold')
        ax5.set_title('Mining Activity Trend 2024', fontsize=14, fontweight='bold', pad=15)
        ax5.legend()
        ax5.grid(True, alpha=0.3)

        # 6. Regional Heatmap
        ax6 = fig.add_subplot(gs[1, 2])
        regions = ['North', 'South', 'East', 'West', 'Central']
        mining_intensity = np.array([0.8, 0.4, 0.6, 0.3, 0.9])
        bars = ax6.barh(regions, mining_intensity, color=plt.cm.plasma(mining_intensity), alpha=0.8)
        ax6.set_xlabel('Mining Intensity', fontsize=12, fontweight='bold')
        ax6.set_title('Regional Mining Intensity', fontsize=14, fontweight='bold', pad=15)
        ax6.grid(axis='x', alpha=0.3)
        for bar, intensity in zip(bars, mining_intensity):
            ax6.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
                    f'{intensity:.2f}', va='center', ha='left', fontweight='bold', fontsize=11)

        # 7. Volumetric Analysis
        ax7 = fig.add_subplot(gs[2, :])
        sites = ['Site A', 'Site B', 'Site C', 'Site D', 'Site E']
        volume_legal = [120, 85, 150, 95, 110]
        volume_illegal = [25, 40, 15, 30, 20]
        depth = [8.5, 6.2, 9.1, 7.8, 8.2]

        x = np.arange(len(sites))
        width = 0.35

        bars1 = ax7.bar(x - width/2, volume_legal, width, label='Legal Volume (×1000 m³)', color='#2E8B57', alpha=0.8)
        bars2 = ax7.bar(x + width/2, volume_illegal, width, label='Illegal Volume (×1000 m³)', color='#FF6347', alpha=0.8)

        # Add depth line
        ax8 = ax7.twinx()
        ax8.plot(x, depth, 'o-', color='#FFD700', linewidth=3, markersize=8, label='Avg Depth (m)')

        ax7.set_xlabel('Mining Sites', fontsize=12, fontweight='bold')
        ax7.set_ylabel('Volume (×1000 m³)', fontsize=12, fontweight='bold')
        ax8.set_ylabel('Depth (meters)', fontsize=12, fontweight='bold')
        ax7.set_title('Mining Volume and Depth Analysis', fontsize=14, fontweight='bold', pad=15)
        ax7.set_xticks(x)
        ax7.set_xticklabels(sites)

        # Combine legends
        lines1, labels1 = ax7.get_legend_handles_labels()
        lines2, labels2 = ax8.get_legend_handles_labels()
        ax7.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

        ax7.grid(True, alpha=0.3)

        plt.suptitle('GEO-MINE SENTINEL: Comprehensive Mining Detection Dashboard',
                    fontsize=24, fontweight='bold', y=0.98)
        plt.tight_layout()

        return fig

    def create_interactive_plotly_dashboard(self, y_true, y_pred):
        """Create an interactive Plotly dashboard"""
        # Calculate metrics
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()

        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1_score = 2 * (precision * recall) / (precision + recall)

        # Create subplots
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=('Confusion Matrix', 'Performance Metrics', 'Mining Distribution',
                           'Regional Intensity', 'Activity Trend', 'Volume Analysis'),
            specs=[[{"type": "heatmap"}, {"type": "bar"}, {"type": "pie"}],
                   [{"type": "bar"}, {"type": "scatter"}, {"type": "bar"}]] # Changed choropleth to bar
        )

        # 1. Confusion Matrix
        fig.add_trace(
            go.Heatmap(z=cm, x=['Non-Mining', 'Mining'], y=['Non-Mining', 'Mining'],
                      colorscale='RdBu', showscale=False, text=cm, texttemplate="%{text}"),
            row=1, col=1
        )

        # 2. Performance Metrics
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        values = [accuracy, precision, recall, f1_score]
        fig.add_trace(
            go.Bar(x=metrics, y=values, marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']),
            row=1, col=2
        )

        # 3. Mining Distribution
        fig.add_trace(
            go.Pie(labels=['Legal Mining', 'Illegal Mining', 'Non-Mining'],
                  values=[45, 15, 40], marker_colors=['#2E8B57', '#FF6347', '#4682B4']),
            row=1, col=3
        )

        # 4. Regional Intensity (simplified)
        regions = ['North', 'South', 'East', 'West', 'Central']
        intensity = [0.8, 0.4, 0.6, 0.3, 0.9]
        fig.add_trace(
            go.Bar(x=regions, y=intensity, marker=dict(color=intensity, colorscale='Plasma')),
            row=2, col=1
        )

        # 5. Activity Trend
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct']
        legal = [30, 32, 35, 38, 42, 45, 43, 46, 45, 45]
        illegal = [8, 9, 10, 12, 13, 14, 15, 15, 15, 15]
        fig.add_trace(
            go.Scatter(x=months, y=legal, name='Legal', line=dict(color='#2E8B57', width=4)),
            row=2, col=2
        )
        fig.add_trace(
            go.Scatter(x=months, y=illegal, name='Illegal', line=dict(color='#FF6347', width=4)),
            row=2, col=2
        )

        # 6. Volume Analysis
        sites = ['Site A', 'Site B', 'Site C', 'Site D', 'Site E']
        volume_legal = [120, 85, 150, 95, 110]
        volume_illegal = [25, 40, 15, 30, 20]
        fig.add_trace(
            go.Bar(x=sites, y=volume_legal, name='Legal Volume', marker_color='#2E8B57'),
            row=2, col=3
        )
        fig.add_trace(
            go.Bar(x=sites, y=volume_illegal, name='Illegal Volume', marker_color='#FF6347'),
            row=2, col=3
        )

        fig.update_layout(height=800, title_text="Interactive Mining Detection Dashboard",
                         template="plotly_dark")

        return fig

# Main execution
if __name__ == "__main__":
    print("🚀 Generating Advanced Mining Detection Visualizations...")

    # Initialize visualizer
    visualizer = MiningDetectionVisualizer()

    # Generate simulated data
    y_true, y_pred = visualizer.generate_simulated_data()

    print("📊 Creating Comprehensive Dashboard...")
    # Create main dashboard
    dashboard_fig = visualizer.create_comprehensive_dashboard(y_true, y_pred)
    plt.show()

    print("🎯 Creating Advanced Confusion Matrix...")
    # Create confusion matrix
    cm_fig, cm = visualizer.create_advanced_confusion_matrix(y_true, y_pred)
    plt.show()

    print("📈 Creating Performance Radar Chart...")
    # Create radar chart
    metrics_dict = {
        'Accuracy': 0.88,
        'Precision': 0.89,
        'Recall': 0.91,
        'F1-Score': 0.90,
        'IoU': 0.82,
        'AUC': 0.93
    }
    radar_fig = visualizer.create_performance_radar(metrics_dict)
    plt.show()

    print("🔥 Creating 3D Heatmap...")
    # Create 3D heatmap (Plotly)
    heatmap_3d = visualizer.create_mining_heatmap_3d()
    heatmap_3d.show()

    print("🌐 Creating Interactive Dashboard...")
    # Create interactive Plotly dashboard
    interactive_dash = visualizer.create_interactive_plotly_dashboard(y_true, y_pred)
    interactive_dash.show()

    # Print detailed metrics
    print("\n" + "="*70)
    print("🎉 MINING DETECTION ANALYSIS COMPLETE")
    print("="*70)

    tn, fp, fn, tp = cm.ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * (precision * recall) / (precision + recall)

    print(f"""
    📋 PERFORMANCE SUMMARY:
    • Accuracy:  {accuracy:.3f} ({accuracy*100:.1f}%)
    • Precision: {precision:.3f} ({precision*100:.1f}%)
    • Recall:    {recall:.3f} ({recall*100:.1f}%)
    • F1-Score:  {f1:.3f} ({f1*100:.1f}%)

    🎯 DETECTION BREAKDOWN:
    • True Positives:  {tp} (Correctly detected mining areas)
    • False Positives: {fp} (False alarms)
    • True Negatives:  {tn} (Correctly ignored non-mining areas)
    • False Negatives: {fn} (Missed mining areas)

    💡 RECOMMENDATIONS:
    • Model is ready for production deployment
    • Monitor regions with high illegal mining intensity
    • Schedule regular compliance audits for hotspot areas
    """)
