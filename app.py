import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display, clear_output

# ===========================
# 1️⃣ Load Dataset
# ===========================
url = "https://raw.githubusercontent.com/cr1825/ai-project-data-sheet/main/EdgeAI_Robot_Motion_Control_Dataset.csv"
data = pd.read_csv(url)

print("✅ Data Loaded Successfully")
print("Shape:", data.shape)
print("Columns:", data.columns.tolist())

# Check for Motion_Type column
if "Motion_Type" not in data.columns:
    print("\n⚠️ No Motion_Type column found. Please verify the dataset structure.")
    print("Available columns:", data.columns.tolist())
else:
    print(f"\n📋 Available Motion Types: {data['Motion_Type'].unique()}")
    print(f"Motion Type Distribution:\n{data['Motion_Type'].value_counts()}")

# ===========================
# 2️⃣ Separate Data by Motion Type
# ===========================
# Model 1: Pick & Place → Position Accuracy
pick_place_data = data[data['Motion_Type'].isin(['Pick', 'Place'])].copy()

# Model 2: Weld & Inspect → Velocity Accuracy
weld_inspect_data = data[data['Motion_Type'].isin(['Weld', 'Inspect'])].copy()

print(f"\n📊 Data Distribution:")
print(f"  • Pick/Place samples: {len(pick_place_data)}")
print(f"  • Weld/Inspect samples: {len(weld_inspect_data)}")

# ===========================
# 3️⃣ MODEL 1: Position Accuracy for Pick & Place
# ===========================
print("\n" + "="*60)
print("🎯 MODEL 1: POSITION ACCURACY (Pick & Place Tasks)")
print("="*60)

# Prepare features and target
X_position = pick_place_data[['Desired_Position', 'Desired_Velocity', 'Joint_ID']]
y_position = pick_place_data['Actual_Position']

# Train-test split
X_pos_train, X_pos_test, y_pos_train, y_pos_test = train_test_split(
    X_position, y_position, test_size=0.2, random_state=42
)

# Train Random Forest for Position
print("\n🚀 Training Position Prediction Model...")
rf_position = RandomForestRegressor(
    n_estimators=300, 
    max_depth=25, 
    random_state=42, 
    n_jobs=-1
)
rf_position.fit(X_pos_train, y_pos_train)

# Predictions
y_pos_train_pred = rf_position.predict(X_pos_train)
y_pos_test_pred = rf_position.predict(X_pos_test)

# Metrics for Position Model
r2_pos_train = r2_score(y_pos_train, y_pos_train_pred)
r2_pos_test = r2_score(y_pos_test, y_pos_test_pred)
rmse_pos_train = np.sqrt(mean_squared_error(y_pos_train, y_pos_train_pred))
rmse_pos_test = np.sqrt(mean_squared_error(y_pos_test, y_pos_test_pred))
mae_pos_test = np.mean(np.abs(y_pos_test - y_pos_test_pred))

print("\n📈 Position Model Performance:")
print(f"  Training Set:")
print(f"    • R² Score: {r2_pos_train:.4f}")
print(f"    • RMSE: {rmse_pos_train:.4f}")
print(f"\n  Test Set:")
print(f"    • R² Score: {r2_pos_test:.4f}")
print(f"    • RMSE: {rmse_pos_test:.4f}")
print(f"    • MAE: {mae_pos_test:.4f}")

# Feature importance for Position Model
feature_importance_pos = pd.DataFrame({
    'Feature': X_position.columns,
    'Importance': rf_position.feature_importances_
}).sort_values('Importance', ascending=False)
print("\n🔍 Feature Importance (Position Model):")
print(feature_importance_pos.to_string(index=False))

# ===========================
# 4️⃣ MODEL 2: Velocity Accuracy for Weld & Inspect
# ===========================
print("\n" + "="*60)
print("⚡ MODEL 2: VELOCITY ACCURACY (Weld & Inspect Tasks)")
print("="*60)

# Prepare features and target
X_velocity = weld_inspect_data[['Desired_Position', 'Desired_Velocity', 'Joint_ID']]
y_velocity = weld_inspect_data['Actual_Velocity']

# Train-test split
X_vel_train, X_vel_test, y_vel_train, y_vel_test = train_test_split(
    X_velocity, y_velocity, test_size=0.2, random_state=42
)

# Train Random Forest for Velocity
print("\n🚀 Training Velocity Prediction Model...")
rf_velocity = RandomForestRegressor(
    n_estimators=300, 
    max_depth=25, 
    random_state=42, 
    n_jobs=-1
)
rf_velocity.fit(X_vel_train, y_vel_train)

# Predictions
y_vel_train_pred = rf_velocity.predict(X_vel_train)
y_vel_test_pred = rf_velocity.predict(X_vel_test)

# Metrics for Velocity Model
r2_vel_train = r2_score(y_vel_train, y_vel_train_pred)
r2_vel_test = r2_score(y_vel_test, y_vel_test_pred)
rmse_vel_train = np.sqrt(mean_squared_error(y_vel_train, y_vel_train_pred))
rmse_vel_test = np.sqrt(mean_squared_error(y_vel_test, y_vel_test_pred))
mae_vel_test = np.mean(np.abs(y_vel_test - y_vel_test_pred))

print("\n📈 Velocity Model Performance:")
print(f"  Training Set:")
print(f"    • R² Score: {r2_vel_train:.4f}")
print(f"    • RMSE: {rmse_vel_train:.4f}")
print(f"\n  Test Set:")
print(f"    • R² Score: {r2_vel_test:.4f}")
print(f"    • RMSE: {rmse_vel_test:.4f}")
print(f"    • MAE: {mae_vel_test:.4f}")

# Feature importance for Velocity Model
feature_importance_vel = pd.DataFrame({
    'Feature': X_velocity.columns,
    'Importance': rf_velocity.feature_importances_
}).sort_values('Importance', ascending=False)
print("\n🔍 Feature Importance (Velocity Model):")
print(feature_importance_vel.to_string(index=False))

# ===========================
# 5️⃣ Comparative Summary
# ===========================
print("\n" + "="*60)
print("📊 COMPARATIVE MODEL SUMMARY")
print("="*60)

summary_table = pd.DataFrame({
    'Model': ['Position (Pick/Place)', 'Velocity (Weld/Inspect)'],
    'Train_R2': [r2_pos_train, r2_vel_train],
    'Test_R2': [r2_pos_test, r2_vel_test],
    'Train_RMSE': [rmse_pos_train, rmse_vel_train],
    'Test_RMSE': [rmse_pos_test, rmse_vel_test],
    'Test_MAE': [mae_pos_test, mae_vel_test],
    'Samples': [len(pick_place_data), len(weld_inspect_data)]
})

print("\n" + summary_table.to_string(index=False))

# ===========================
# 6️⃣ Visualization
# ===========================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Position Model - Actual vs Predicted
axes[0, 0].scatter(y_pos_test, y_pos_test_pred, alpha=0.5, s=20)
axes[0, 0].plot([y_pos_test.min(), y_pos_test.max()], 
                [y_pos_test.min(), y_pos_test.max()], 'r--', lw=2)
axes[0, 0].set_xlabel('Actual Position', fontsize=11)
axes[0, 0].set_ylabel('Predicted Position', fontsize=11)
axes[0, 0].set_title(f'Position Model (Pick/Place)\nR²={r2_pos_test:.4f}', 
                     fontsize=12, fontweight='bold')
axes[0, 0].grid(alpha=0.3)

# Plot 2: Position Model - Residuals
residuals_pos = y_pos_test - y_pos_test_pred
axes[0, 1].scatter(y_pos_test_pred, residuals_pos, alpha=0.5, s=20)
axes[0, 1].axhline(y=0, color='r', linestyle='--', lw=2)
axes[0, 1].set_xlabel('Predicted Position', fontsize=11)
axes[0, 1].set_ylabel('Residuals', fontsize=11)
axes[0, 1].set_title(f'Position Model Residuals\nRMSE={rmse_pos_test:.4f}', 
                     fontsize=12, fontweight='bold')
axes[0, 1].grid(alpha=0.3)

# Plot 3: Velocity Model - Actual vs Predicted
axes[1, 0].scatter(y_vel_test, y_vel_test_pred, alpha=0.5, s=20, color='orange')
axes[1, 0].plot([y_vel_test.min(), y_vel_test.max()], 
                [y_vel_test.min(), y_vel_test.max()], 'r--', lw=2)
axes[1, 0].set_xlabel('Actual Velocity', fontsize=11)
axes[1, 0].set_ylabel('Predicted Velocity', fontsize=11)
axes[1, 0].set_title(f'Velocity Model (Weld/Inspect)\nR²={r2_vel_test:.4f}', 
                     fontsize=12, fontweight='bold')
axes[1, 0].grid(alpha=0.3)

# Plot 4: Velocity Model - Residuals
residuals_vel = y_vel_test - y_vel_test_pred
axes[1, 1].scatter(y_vel_test_pred, residuals_vel, alpha=0.5, s=20, color='orange')
axes[1, 1].axhline(y=0, color='r', linestyle='--', lw=2)
axes[1, 1].set_xlabel('Predicted Velocity', fontsize=11)
axes[1, 1].set_ylabel('Residuals', fontsize=11)
axes[1, 1].set_title(f'Velocity Model Residuals\nRMSE={rmse_vel_test:.4f}', 
                     fontsize=12, fontweight='bold')
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.show()

# ===========================
# 7️⃣ Time Calculation Functions
# ===========================

def calculate_movement_time(current_pos, target_pos, velocity, acceleration=10.0, deceleration=10.0):
    """
    Calculate time for a joint to move from current to target position
    Uses trapezoidal velocity profile with acceleration, constant velocity, and deceleration phases
    
    Parameters:
    - current_pos: Starting position
    - target_pos: Target position
    - velocity: Maximum velocity
    - acceleration: Acceleration rate (default: 10 degrees/s²)
    - deceleration: Deceleration rate (default: 10 degrees/s²)
    
    Returns:
    - total_time: Time in seconds
    - phases: Dictionary with time breakdown
    """
    distance = abs(target_pos - current_pos)
    
    # Time to reach max velocity
    t_accel = velocity / acceleration
    t_decel = velocity / deceleration
    
    # Distance covered during acceleration and deceleration
    d_accel = 0.5 * acceleration * t_accel**2
    d_decel = 0.5 * deceleration * t_decel**2
    
    # Check if we reach max velocity
    if (d_accel + d_decel) >= distance:
        # Triangular profile (never reach max velocity)
        # v_peak = sqrt(2 * distance * acceleration * deceleration / (acceleration + deceleration))
        v_peak = np.sqrt(2 * distance * acceleration * deceleration / (acceleration + deceleration))
        t_accel = v_peak / acceleration
        t_decel = v_peak / deceleration
        t_constant = 0
    else:
        # Trapezoidal profile
        d_constant = distance - d_accel - d_decel
        t_constant = d_constant / velocity
    
    total_time = t_accel + t_constant + t_decel
    
    phases = {
        'acceleration': t_accel,
        'constant_velocity': t_constant,
        'deceleration': t_decel
    }
    
    return total_time, phases

def calculate_task_time(joint_movements, task_type, overlap_factor=0.7):
    """
    Calculate total task time considering all joint movements
    
    Parameters:
    - joint_movements: List of dictionaries containing joint movement details
    - task_type: 'Pick & Place' or 'Weld & Inspect'
    - overlap_factor: How much joints can move simultaneously (0=sequential, 1=fully parallel)
    
    Returns:
    - total_time: Total task time
    - breakdown: Detailed time breakdown
    """
    movement_times = []
    
    for movement in joint_movements:
        time, phases = calculate_movement_time(
            movement['current_pos'],
            movement['target_pos'],
            movement['velocity']
        )
        movement_times.append({
            'joint_id': movement['joint_id'],
            'time': time,
            'phases': phases,
            'distance': abs(movement['target_pos'] - movement['current_pos'])
        })
    
    # Maximum time (longest joint movement)
    max_time = max([m['time'] for m in movement_times]) if movement_times else 0
    
    # Total sequential time
    total_sequential = sum([m['time'] for m in movement_times])
    
    # Actual time considering parallel movement
    actual_movement_time = max_time + (1 - overlap_factor) * (total_sequential - max_time)
    
    # Add task-specific overhead times
    if task_type == 'Pick & Place':
        gripper_time = 0.5  # Time to open/close gripper
        settling_time = 0.2  # Time to settle at position
        total_overhead = gripper_time + settling_time
    else:  # Weld & Inspect
        setup_time = 0.3  # Setup time for welding/inspection
        process_time = 1.0  # Actual welding or inspection time
        total_overhead = setup_time + process_time
    
    total_time = actual_movement_time + total_overhead
    
    breakdown = {
        'movement_time': actual_movement_time,
        'overhead_time': total_overhead,
        'max_joint_time': max_time,
        'sequential_time': total_sequential,
        'overlap_factor': overlap_factor,
        'joint_details': movement_times
    }
    
    return total_time, breakdown

# ===========================
# 8️⃣ Enhanced Interactive Widget Interface with Time Estimation
# ===========================
print("\n" + "="*60)
print("🤖 INTERACTIVE ROBOT MOTION PREDICTION INTERFACE")
print("="*60)

# Get available joints
available_joints = sorted(data['Joint_ID'].unique())

# Output widget for displaying results
output = widgets.Output()

# Store joint movements
joint_movements_store = []

# Main interface widgets
interface_title = widgets.HTML(
    value="<h2 style='color: #1976D2; text-align: center;'>🤖 Robot Motion Control Predictor with Time Estimation</h2>"
)

task_type = widgets.Dropdown(
    options=['Pick & Place', 'Weld & Inspect'],
    value='Pick & Place',
    description='Task Type:',
    style={'description_width': '150px'},
    layout=widgets.Layout(width='400px')
)

prediction_type = widgets.Dropdown(
    options=['Position Prediction', 'Velocity Prediction'],
    value='Position Prediction',
    description='Prediction Type:',
    style={'description_width': '150px'},
    layout=widgets.Layout(width='400px'),
    disabled=True
)

joint_id = widgets.Dropdown(
    options=available_joints,
    value=available_joints[0],
    description='Joint ID:',
    style={'description_width': '150px'},
    layout=widgets.Layout(width='400px')
)

current_position = widgets.FloatText(
    value=0.0,
    description='Current Position:',
    style={'description_width': '150px'},
    layout=widgets.Layout(width='400px')
)

desired_position = widgets.FloatText(
    value=45.0,
    description='Desired Position:',
    style={'description_width': '150px'},
    layout=widgets.Layout(width='400px')
)

desired_velocity = widgets.FloatText(
    value=2.0,
    description='Desired Velocity:',
    style={'description_width': '150px'},
    layout=widgets.Layout(width='400px')
)

add_joint_button = widgets.Button(
    description='➕ Add Joint Movement',
    button_style='info',
    icon='plus',
    layout=widgets.Layout(width='200px', height='35px')
)

clear_joints_button = widgets.Button(
    description='🗑️ Clear All Joints',
    button_style='warning',
    icon='trash',
    layout=widgets.Layout(width='200px', height='35px')
)

calculate_button = widgets.Button(
    description='⏱️ Calculate Total Time',
    button_style='success',
    icon='clock-o',
    layout=widgets.Layout(width='200px', height='40px')
)

joints_display = widgets.HTML(
    value="<div style='background-color: #F5F5F5; padding: 10px; border-radius: 5px; min-height: 50px;'>"
          "<b>📋 Joint Movements Queue:</b> (Empty)</div>"
)

info_box = widgets.HTML(
    value="<div style='background-color: #E3F2FD; padding: 10px; border-radius: 5px; margin: 10px 0;'>"
          "<b>ℹ️ Info:</b> Pick & Place tasks predict <b>Position</b>, "
          "while Weld & Inspect tasks predict <b>Velocity</b>. Add multiple joints to calculate total task time.</div>"
)

# Update prediction type based on task type
def update_prediction_type(change):
    global joint_movements_store
    joint_movements_store = []
    update_joints_display()
    
    if change['new'] == 'Pick & Place':
        prediction_type.value = 'Position Prediction'
        prediction_type.options = ['Position Prediction']
        current_position.value = 0.0
        desired_position.value = 45.0
        desired_velocity.value = 2.0
        info_box.value = ("<div style='background-color: #E3F2FD; padding: 10px; border-radius: 5px; margin: 10px 0;'>"
                         "<b>🎯 Pick & Place:</b> This task focuses on accurate <b>positioning</b> "
                         "for grasping and placing objects. Add joints to calculate movement time.</div>")
    else:  # Weld & Inspect
        prediction_type.value = 'Velocity Prediction'
        prediction_type.options = ['Velocity Prediction']
        current_position.value = 0.0
        desired_position.value = 40.0
        desired_velocity.value = 1.8
        info_box.value = ("<div style='background-color: #FFF3E0; padding: 10px; border-radius: 5px; margin: 10px 0;'>"
                         "<b>⚡ Weld & Inspect:</b> This task requires precise <b>velocity control</b> "
                         "for smooth welding and inspection movements. Add joints to calculate movement time.</div>")

task_type.observe(update_prediction_type, names='value')

def update_joints_display():
    """Update the display of added joints"""
    if not joint_movements_store:
        joints_display.value = ("<div style='background-color: #F5F5F5; padding: 10px; border-radius: 5px; min-height: 50px;'>"
                               "<b>📋 Joint Movements Queue:</b> (Empty)</div>")
    else:
        joints_html = "<div style='background-color: #E8F5E9; padding: 10px; border-radius: 5px;'>"
        joints_html += "<b>📋 Joint Movements Queue:</b><br><br>"
        for i, jm in enumerate(joint_movements_store, 1):
            joints_html += f"<b>{i}. Joint {jm['joint_id']}</b>: "
            joints_html += f"{jm['current_pos']:.2f}° → {jm['target_pos']:.2f}° "
            joints_html += f"@ {jm['velocity']:.2f}°/s<br>"
        joints_html += "</div>"
        joints_display.value = joints_html

def add_joint_movement(b):
    """Add a joint movement to the queue"""
    global joint_movements_store
    
    with output:
        clear_output(wait=True)
        
        # Make prediction first
        if task_type.value == 'Pick & Place':
            input_data = pd.DataFrame({
                'Desired_Position': [desired_position.value],
                'Desired_Velocity': [desired_velocity.value],
                'Joint_ID': [joint_id.value]
            })
            predicted_pos = rf_position.predict(input_data)[0]
            predicted_vel = desired_velocity.value  # Use desired velocity
        else:
            input_data = pd.DataFrame({
                'Desired_Position': [desired_position.value],
                'Desired_Velocity': [desired_velocity.value],
                'Joint_ID': [joint_id.value]
            })
            predicted_vel = rf_velocity.predict(input_data)[0]
            predicted_pos = desired_position.value  # Use desired position
        
        # Add to store
        movement = {
            'joint_id': joint_id.value,
            'current_pos': current_position.value,
            'target_pos': predicted_pos,
            'velocity': abs(predicted_vel) if predicted_vel != 0 else 1.0
        }
        joint_movements_store.append(movement)
        
        update_joints_display()
        
        print(f"✅ Added Joint {joint_id.value} movement to queue!")
        print(f"   Current: {current_position.value:.2f}° → Target: {predicted_pos:.2f}°")
        print(f"   Velocity: {movement['velocity']:.2f}°/s")
        print(f"\n📊 Total joints in queue: {len(joint_movements_store)}")

def clear_joints(b):
    """Clear all joint movements"""
    global joint_movements_store
    joint_movements_store = []
    update_joints_display()
    
    with output:
        clear_output(wait=True)
        print("🗑️ All joint movements cleared!")

def calculate_total_time(b):
    """Calculate total task time"""
    with output:
        clear_output(wait=True)
        
        if not joint_movements_store:
            print("❌ No joint movements added! Please add at least one joint movement.")
            return
        
        print("\n" + "="*70)
        print("⏱️  TASK TIME ESTIMATION REPORT")
        print("="*70)
        
        # Calculate total time
        total_time, breakdown = calculate_task_time(
            joint_movements_store,
            task_type.value,
            overlap_factor=0.7
        )
        
        print(f"\n📋 TASK DETAILS:")
        print(f"  • Task Type: {task_type.value}")
        print(f"  • Number of Joints: {len(joint_movements_store)}")
        print(f"  • Parallel Execution: 70% (joints move simultaneously)")
        
        print(f"\n⏱️  TIME BREAKDOWN:")
        print(f"  • Pure Movement Time: {breakdown['movement_time']:.3f} seconds")
        print(f"  • Task Overhead Time: {breakdown['overhead_time']:.3f} seconds")
        print(f"  • Longest Joint Movement: {breakdown['max_joint_time']:.3f} seconds")
        print(f"  • Sequential Time (if no overlap): {breakdown['sequential_time']:.3f} seconds")
        
        print(f"\n🎯 TOTAL ESTIMATED TIME: {total_time:.3f} seconds ({total_time/60:.2f} minutes)")
        
        print(f"\n📊 INDIVIDUAL JOINT MOVEMENTS:")
        print("-" * 70)
        for i, jm in enumerate(breakdown['joint_details'], 1):
            print(f"\n{i}. Joint {joint_movements_store[i-1]['joint_id']}:")
            print(f"   • Movement: {joint_movements_store[i-1]['current_pos']:.2f}° → "
                  f"{joint_movements_store[i-1]['target_pos']:.2f}°")
            print(f"   • Distance: {jm['distance']:.2f}°")
            print(f"   • Velocity: {joint_movements_store[i-1]['velocity']:.2f}°/s")
            print(f"   • Time: {jm['time']:.3f}s "
                  f"(Accel: {jm['phases']['acceleration']:.3f}s, "
                  f"Constant: {jm['phases']['constant_velocity']:.3f}s, "
                  f"Decel: {jm['phases']['deceleration']:.3f}s)")
        
        print("\n" + "="*70)
        
        # Time efficiency analysis
        efficiency = (breakdown['max_joint_time'] / breakdown['sequential_time']) * 100
        print(f"\n📈 EFFICIENCY ANALYSIS:")
        print(f"  • Time saved by parallel execution: "
              f"{breakdown['sequential_time'] - breakdown['movement_time']:.3f} seconds")
        print(f"  • Efficiency gain: {100 - efficiency:.1f}%")
        
        # Performance rating
        if total_time < 2.0:
            rating = "⚡ VERY FAST"
        elif total_time < 4.0:
            rating = "✅ FAST"
        elif total_time < 6.0:
            rating = "✓ MODERATE"
        elif total_time < 10.0:
            rating = "⚠️ SLOW"
        else:
            rating = "❌ VERY SLOW"
        
        print(f"\n🏆 Performance Rating: {rating}")

add_joint_button.on_click(add_joint_movement)
clear_joints_button.on_click(clear_joints)
calculate_button.on_click(calculate_total_time)

# Layout
button_row = widgets.HBox([add_joint_button, clear_joints_button], 
                          layout=widgets.Layout(justify_content='space-between'))

input_section = widgets.VBox([
    interface_title,
    info_box,
    task_type,
    prediction_type,
    joint_id,
    current_position,
    desired_position,
    desired_velocity,
    button_row,
    joints_display,
    calculate_button
], layout=widgets.Layout(
    padding='20px',
    border='3px solid #1976D2',
    border_radius='10px',
    width='450px'
))

main_interface = widgets.VBox([
    input_section,
    output
], layout=widgets.Layout(padding='10px'))

# Display interface
display(main_interface)

print("\n✨ Interactive interface with time estimation ready!")
print("💡 Add multiple joint movements and click 'Calculate Total Time' to see the complete task duration.")