import streamlit as st
import pandas as pd
from datetime import datetime
import os
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from appwrite.client import Client
from appwrite.services.databases import Databases
from appwrite.query import Query
from dotenv import load_dotenv
import platform
import pytz
import logging
import time
import hashlib

# Setup logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Configuration from .env file
CHART_FILE = os.getenv('CHART_FILE', 'attendance_by_college.png')
DEBUG_MODE = os.getenv('DEBUG_MODE', 'True').lower() == 'true'

# Appwrite configuration
APPWRITE_ENDPOINT = os.getenv('APPWRITE_ENDPOINT', 'https://cloud.appwrite.io/v1')
APPWRITE_PROJECT_ID = os.getenv('APPWRITE_PROJECT_ID')
APPWRITE_API_KEY = os.getenv('APPWRITE_API_KEY')
APPWRITE_DATABASE_ID = os.getenv('APPWRITE_DATABASE_ID')
APPWRITE_INTERNS_COLLECTION_ID = os.getenv('APPWRITE_INTERNS_COLLECTION_ID')
APPWRITE_ATTENDANCE_COLLECTION_ID = os.getenv('APPWRITE_ATTENDANCE_COLLECTION_ID')

# Validate Appwrite configuration
if not all([APPWRITE_PROJECT_ID, APPWRITE_API_KEY, APPWRITE_DATABASE_ID, 
            APPWRITE_ATTENDANCE_COLLECTION_ID, APPWRITE_INTERNS_COLLECTION_ID]):
    missing = [k for k, v in {
        'APPWRITE_PROJECT_ID': APPWRITE_PROJECT_ID,
        'APPWRITE_API_KEY': APPWRITE_API_KEY,
        'APPWRITE_DATABASE_ID': APPWRITE_DATABASE_ID,
        'APPWRITE_ATTENDANCE_COLLECTION_ID': APPWRITE_ATTENDANCE_COLLECTION_ID,
        'APPWRITE_INTERNS_COLLECTION_ID': APPWRITE_INTERNS_COLLECTION_ID
    }.items() if not v]
    logger.error(f"Missing Appwrite configuration: {', '.join(missing)}")
    st.error(f"❌️ Missing Appwrite configuration: {', '.join(missing)}")
    st.stop()

# Parse admin credentials
ADMIN_CREDENTIALS = {}
admin_creds_str = os.getenv('ADMIN_CREDENTIALS', '')
if admin_creds_str:
    for cred in admin_creds_str.split(','):
        if ':' in cred:
            username, password = cred.strip().split(':', 1)
            ADMIN_CREDENTIALS[username] = password
    logger.debug(f"Loaded {len(ADMIN_CREDENTIALS)} admin credentials")

# Initialize Appwrite client
client = Client()
client.set_endpoint(APPWRITE_ENDPOINT)
client.set_project(APPWRITE_PROJECT_ID)
client.set_key(APPWRITE_API_KEY)
databases = Databases(client)
logger.debug("Appwrite client initialized")

# Session state initialization
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'user_role' not in st.session_state:
    st.session_state.user_role = None
if 'username' not in st.session_state:
    st.session_state.username = ""
# Generate a better device fingerprint
if 'device_fingerprint' not in st.session_state:
    try:
        # Extract user-agent from request headers (only available in Streamlit Community Cloud / behind web server)
        user_agent = st._runtime.scriptrunner.get_script_run_ctx().request.headers.get("User-Agent", "unknown")
    except Exception:
        user_agent = "unknown"

    raw_device_info = f"{user_agent}_{platform.platform()}"
    st.session_state.device_fingerprint = hashlib.sha256(raw_device_info.encode()).hexdigest()[:16]


# Timezone for IST
IST = pytz.timezone('Asia/Kolkata')

# Debug timezone
def get_current_time_info():
    try:
        local_time = datetime.now()
        ist_time = datetime.now(IST)
        system_tz = time.tzname
        return {
            'local_time': local_time.strftime('%Y-%m-%d %H:%M:%S %Z'),
            'ist_time': ist_time.strftime('%Y-%m-%d %H:%M:%S %Z'),
            'system_tz': system_tz
        }
    except Exception as e:
        logger.error(f"Error in get_current_time_info: {str(e)}", exc_info=True)
        return {
            'local_time': 'Error',
            'ist_time': 'Error',
            'system_tz': 'Error'
        }

# Admin login function
def admin_login():
    st.header('🔐 Admin Login')
    
    with st.form('admin_login'):
        username = st.text_input('👤 Username')
        password = st.text_input('🔑 Password', type='password')
        login_button = st.form_submit_button('Login')
        
        if login_button:
            if username in ADMIN_CREDENTIALS and ADMIN_CREDENTIALS[username] == password:
                st.session_state.authenticated = True
                st.session_state.user_role = username
                logger.info(f"Admin {username} logged in successfully")
                st.success('✅ Login successful!')
                st.rerun()
            else:
                st.error('❌ Invalid credentials')
                logger.warning(f"Failed login attempt: {username}")

# Verify username in Appwrite interns collection
def verify_username(username):
    logger.debug(f"Verifying username: {username}")
    try:
        result = databases.list_documents(
            database_id=APPWRITE_DATABASE_ID,
            collection_id=APPWRITE_INTERNS_COLLECTION_ID,
            queries=[Query.equal('username', username)]
        )
        logger.debug(f"Query result: {result['total']} documents found")
        if result['total'] > 0:
            college_name = result['documents'][0].get('college_name', '')
            logger.debug(f"Username {username} verified, college: {college_name}")
            return college_name
        st.error(f"❌ Username '{username}' not found.")
        logger.warning(f"Username {username} not found")
        return None
    except Exception as e:
        st.error(f"Error verifying username: {str(e)}")
        logger.error(f"Error verifying username {username}: {str(e)}", exc_info=True)
        return None

# Check if device has already marked attendance today
def check_device_attendance_today(username, action):
    logger.debug(f"Checking device attendance for {username}, action: {action}")
    today = datetime.now(IST).strftime("%Y-%m-%d")
    device_id = st.session_state.device_fingerprint
    
    try:
        # Check if this device has already marked the same action today
        result = databases.list_documents(
            database_id=APPWRITE_DATABASE_ID,
            collection_id=APPWRITE_ATTENDANCE_COLLECTION_ID,
            queries=[
                Query.equal('username', username),
                Query.equal('date', today),
                Query.equal('device_id', device_id)
            ]
        )
        
        existing_records = result['documents']
        if existing_records:
            record = existing_records[0]
            if action == "In" and record.get('in_time'):
                return False, "You have already marked IN from this device today."
            elif action == "Out" and record.get('out_time'):
                return False, "You have already marked OUT from this device today."
        
        return True, "OK"
    except Exception as e:
        logger.error(f"Error checking device attendance: {str(e)}", exc_info=True)
        return True, "OK"  # Allow if check fails

# Import CSV to interns collection
def import_interns_csv(uploaded_file):
    logger.debug("Importing interns CSV")
    try:
        df = pd.read_csv(uploaded_file)
        if 'username' not in df.columns or 'college_name' not in df.columns:
            st.error("❌ CSV must have 'username' and 'college_name' columns")
            logger.error("Invalid CSV format: missing required columns")
            return 0
        
        existing = databases.list_documents(
            database_id=APPWRITE_DATABASE_ID,
            collection_id=APPWRITE_INTERNS_COLLECTION_ID,
            queries=[Query.limit(1000)]
        )
        existing_usernames = {doc['username'] for doc in existing['documents']}
        logger.debug(f"Found {len(existing_usernames)} existing usernames")
        
        success_count = 0
        for _, row in df.iterrows():
            username = str(row['username']).strip()
            college_name = str(row['college_name']).strip() if pd.notna(row['college_name']) else ''
            
            if not username:
                logger.warning(f"Skipping empty username at row {_+2}")
                continue
                
            if username in existing_usernames:
                logger.debug(f"Skipping existing username: {username}")
                continue
                
            try:
                databases.create_document(
                    database_id=APPWRITE_DATABASE_ID,
                    collection_id=APPWRITE_INTERNS_COLLECTION_ID,
                    document_id='unique()',
                    data={
                        'username': username,
                        'college_name': college_name
                    }
                )
                success_count += 1
                logger.debug(f"Added intern: {username}")
            except Exception as e:
                st.warning(f"Skipped {username}: {str(e)}")
                logger.warning(f"Skipped {username}: {str(e)}")
        
        logger.info(f"Imported {success_count} interns")
        return success_count
    except Exception as e:
        st.error(f"Error importing CSV: {str(e)}")
        logger.error(f"Error importing CSV: {str(e)}", exc_info=True)
        return 0

# Load attendance records from Appwrite
def load_records():
    logger.debug("Loading attendance records")
    try:
        result = databases.list_documents(
            database_id=APPWRITE_DATABASE_ID,
            collection_id=APPWRITE_ATTENDANCE_COLLECTION_ID,
            queries=[Query.limit(1000)]
        )
        logger.debug(f"Loaded {result['total']} attendance records")
        records = result['documents']
        if not records:
            if DEBUG_MODE:
                st.info("No attendance records found in Appwrite.")
            return pd.DataFrame(columns=[
                'username', 'college_name', 'date', 'in_time', 'out_time',
                'total_hours', 'status', 'device_id'
            ])
        df = pd.DataFrame([{
            'username': r.get('username', ''),
            'college_name': r.get('college_name', ''),
            'date': r.get('date', ''),
            'in_time': r.get('in_time', ''),
            'out_time': r.get('out_time', ''),
            'total_hours': float(r.get('total_hours', 0.0)),
            'status': r.get('status', ''),
            'device_id': r.get('device_id', '')
        } for r in result['documents']])
        
        # Ensure all columns exist
        required_columns = ['username', 'college_name', 'date', 'in_time', 'out_time', 
                           'total_hours', 'status', 'device_id']
        for col in required_columns:
            if col not in df.columns:
                if col == 'total_hours':
                    df[col] = 0.0
                else:
                    df[col] = ''
        return df
    except Exception as e:
        st.error(f"Error loading records: {str(e)}")
        logger.error(f"Error loading records: {str(e)}", exc_info=True)
        return pd.DataFrame(columns=[
            'username', 'college_name', 'date', 'in_time', 'out_time',
            'total_hours', 'status', 'device_id'
        ])
def save_record(username, college_name, action):
    logger.debug(f"Saving record: username={username}, action={action}, device_id={st.session_state.device_fingerprint}")

    today = datetime.now(IST).strftime("%Y-%m-%d")
    current_time = datetime.now(IST).strftime("%H:%M:%S")
    device_id = st.session_state.device_fingerprint

    try:
        # Check if this device has already been used by another user today
        device_check = databases.list_documents(
            database_id=APPWRITE_DATABASE_ID,
            collection_id=APPWRITE_ATTENDANCE_COLLECTION_ID,
            queries=[
                Query.equal('device_id', device_id),
                Query.equal('date', today)
            ]
        )
        for record in device_check['documents']:
            if record['username'] != username:
                st.warning(f"⚠️ This device has already been used to mark attendance for {record['username']} on {today}.")
                logger.warning(f"Device {device_id} already used by {record['username']}")
                return False

        # Get user attendance record for today
        username_check = databases.list_documents(
            database_id=APPWRITE_DATABASE_ID,
            collection_id=APPWRITE_ATTENDANCE_COLLECTION_ID,
            queries=[
                Query.equal('username', username),
                Query.equal('date', today)
            ]
        )
        existing_record = username_check['documents'][0] if username_check['documents'] else None

        # Handle IN
        if action == "In":
            if existing_record and existing_record.get('in_time'):
                st.warning(f"⚠️ {username} has already marked IN on {today}.")
                return False

            if existing_record:
                # Update IN
                databases.update_document(
                    database_id=APPWRITE_DATABASE_ID,
                    collection_id=APPWRITE_ATTENDANCE_COLLECTION_ID,
                    document_id=existing_record['$id'],
                    data={
                        'in_time': current_time,
                        'status': 'Checked In',
                        'out_time': '',
                        'total_hours': 0.0,
                        'device_id': device_id
                    }
                )
                st.success(f"✅ Marked IN for {username} at {current_time}")
            else:
                # New IN
                databases.create_document(
                    database_id=APPWRITE_DATABASE_ID,
                    collection_id=APPWRITE_ATTENDANCE_COLLECTION_ID,
                    document_id='unique()',
                    data={
                        'username': username,
                        'college_name': college_name,
                        'date': today,
                        'in_time': current_time,
                        'out_time': '',
                        'total_hours': 0.0,
                        'status': 'Checked In',
                        'device_id': device_id
                    }
                )
                st.success(f"✅ Marked IN for {username} at {current_time}")
            return True

        # Handle OUT
        elif action == "Out":
            if not existing_record:
                st.error(f"❌ {username} has not marked IN today.")
                return False

            if not existing_record.get('in_time'):
                st.error(f"❌ IN time not found for {username}.")
                return False

            if existing_record.get('out_time'):
                st.warning(f"⚠️ {username} has already marked OUT on {today}.")
                return False

            try:
                in_time_dt = datetime.strptime(existing_record['in_time'], "%H:%M:%S")
                out_time_dt = datetime.strptime(current_time, "%H:%M:%S")
                total_hours = round((out_time_dt - in_time_dt).total_seconds() / 3600, 2)
            except Exception as e:
                logger.warning(f"Time calculation error: {str(e)}")
                total_hours = 0.0

            # Update OUT
            databases.update_document(
                database_id=APPWRITE_DATABASE_ID,
                collection_id=APPWRITE_ATTENDANCE_COLLECTION_ID,
                document_id=existing_record['$id'],
                data={
                    'out_time': current_time,
                    'total_hours': total_hours,
                    'status': 'Checked Out',
                    'device_id': device_id
                }
            )
            st.success(f"✅ Marked OUT for {username} at {current_time}")
            return True

        else:
            st.error("Invalid action.")
            return False

    except Exception as e:
        st.error(f"❌ Error saving record: {str(e)}")
        logger.error(f"Exception for {username}: {str(e)}", exc_info=True)
        return False

# Calculate summary statistics
def calculate_summary_stats(df):
    logger.debug("Calculating summary stats")
    if df.empty or 'username' not in df.columns:
        return {
            'total_interns': 0,
            'total_colleges': 0,
            'total_records': 0,
            'active_days': 0,
            'avg_hours': 0,
            'total_hours': 0,
            'complete_sessions': 0,
            'checked_in_today': 0,
            'checked_out_today': 0
        }
    
    today = datetime.now(IST).strftime("%Y-%m-%d")
    today_records = df[df['date'] == today]
    complete_records = df[(df['in_time'] != '') & (df['out_time'] != '') & (df['total_hours'] != '')]
    
    stats = {
        'total_interns': df['username'].nunique(),
        'total_colleges': df['college_name'].nunique() if 'college_name' in df.columns else 0,
        'total_records': len(df),
        'active_days': df['date'].nunique() if 'date' in df.columns else 0,
        'avg_hours': 0,
        'total_hours': 0,
        'complete_sessions': len(complete_records),
        'checked_in_today': len(today_records[today_records['in_time'] != '']),
        'checked_out_today': len(today_records[today_records['out_time'] != ''])
    }
    
    if not complete_records.empty:
        hours_data = pd.to_numeric(complete_records['total_hours'], errors='coerce').dropna()
        if not hours_data.empty:
            stats['avg_hours'] = round(hours_data.mean(), 2)
            stats['total_hours'] = round(hours_data.sum(), 2)
    
    logger.debug(f"Summary stats: {stats}")
    return stats


def generate_enhanced_analytics(df):
    logger.debug("Generating enhanced analytics")
    if df.empty or 'username' not in df.columns:
        st.warning("No valid data to display.")
        return

    # College-wise statistics
    college_stats = df.groupby('college_name').agg({
        'username': 'nunique',
        'date': 'nunique',
        'total_hours': lambda x: pd.to_numeric(x, errors='coerce').sum()
    }).reset_index()
    college_stats.columns = ['College Name', 'Unique Interns', 'Active Days', 'Total Hours']
    college_stats['Total Hours'] = college_stats['Total Hours'].round(2)

    # Daily trends
    daily_trends = df.groupby('date').agg({
        'username': 'nunique',
        'status': lambda x: (x == 'Checked In').sum(),
        'total_hours': lambda x: pd.to_numeric(x, errors='coerce').sum()
    }).reset_index()
    daily_trends.columns = ['Date', 'Unique Interns', 'Check-ins', 'Total Hours']
    daily_trends['Total Hours'] = daily_trends['Total Hours'].round(2)
    daily_trends['Date'] = pd.to_datetime(daily_trends['Date'])

    # Status distribution
    status_counts = df['status'].value_counts().reset_index()
    status_counts.columns = ['Status', 'Count']

    # Subplots
    fig = make_subplots(
        rows=2, cols=2,
        specs=[[{"type": "xy"}, {"type": "domain"}],
               [{"type": "xy"}, {"type": "xy"}]],
        subplot_titles=("👥 Unique Interns by College",
                        "📊 Status Distribution",
                        "📈 Daily Attendance Trends",
                        "⏰ Session Hours Distribution"),
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )

    # Bar chart
    if not college_stats.empty:
        colors = px.colors.qualitative.Set3[:len(college_stats)]
        bar_trace = go.Bar(
            x=college_stats['College Name'],
            y=college_stats['Unique Interns'],
            marker=dict(color=colors, line=dict(color='rgb(8,48,107)', width=1.5), opacity=0.8),
            text=college_stats['Unique Interns'],
            textposition='auto',
            textfont=dict(size=12, color='white'),
            hovertemplate=(
                '<b>%{x}</b><br>' +
                'Unique Interns: %{y}<br>' +
                'Active Days: %{customdata[0]}<br>' +
                'Total Hours: %{customdata[1]:.1f}h<br><extra></extra>'
            ),
            customdata=college_stats[['Active Days', 'Total Hours']].values,
            showlegend=False
        )
        fig.add_trace(bar_trace, row=1, col=1)

    # Pie chart
    if not status_counts.empty:
        pie_trace = go.Pie(
            labels=status_counts['Status'],
            values=status_counts['Count'],
            textinfo='percent+label',
            textfont=dict(size=12),
            marker=dict(colors=px.colors.qualitative.Pastel, line=dict(color='#FFFFFF', width=2)),
            hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>',
            pull=[0.05 if 'In' in status else 0 for status in status_counts['Status']],
            showlegend=True
        )
        fig.add_trace(pie_trace, row=1, col=2)

    # Line chart
    if not daily_trends.empty:
        line1 = go.Scatter(
            x=daily_trends['Date'],
            y=daily_trends['Unique Interns'],
            mode='lines+markers',
            name='Unique Interns',
            line=dict(color='#FF6B6B', width=3),
            marker=dict(size=8, color='#FF6B6B'),
            hovertemplate='<b>%{x|%Y-%m-%d}</b><br>Unique Interns: %{y}<br>Total Hours: %{customdata:.1f}h<extra></extra>',
            customdata=daily_trends['Total Hours']
        )
        line2 = go.Scatter(
            x=daily_trends['Date'],
            y=daily_trends['Check-ins'],
            mode='lines+markers',
            name='Check-ins',
            line=dict(color='#4ECDC4', width=3),
            marker=dict(size=8, color='#4ECDC4'),
            hovertemplate='<b>%{x|%Y-%m-%d}</b><br>Check-ins: %{y}<br>Total Hours: %{customdata:.1f}h<extra></extra>',
            customdata=daily_trends['Total Hours']
        )
        fig.add_trace(line1, row=2, col=1)
        fig.add_trace(line2, row=2, col=1)

    # Histogram
    hours_data = pd.to_numeric(df['total_hours'], errors='coerce').dropna()
    if not hours_data.empty:
        hist = go.Histogram(
            x=hours_data,
            nbinsx=max(10, min(20, len(hours_data) // 3)),
            marker=dict(color='#FFEAA7', line=dict(color='#FDCB6E', width=1), opacity=0.8),
            hovertemplate='Hours Range: %{x}<br>Count: %{y}<extra></extra>',
            showlegend=False
        )
        fig.add_trace(hist, row=2, col=2)

        # Add vertical line for mean
        mean_hours = hours_data.mean()
        fig.add_shape(
            type="line",
            x0=mean_hours, x1=mean_hours,
            y0=0, y1=1,
            xref='x4', yref='paper',
            line=dict(color="red", dash="dash", width=2)
        )
        fig.add_annotation(
            x=mean_hours, y=1.05,
            xref='x4', yref='paper',
            text=f"Mean: {mean_hours:.1f}h",
            showarrow=False,
            font=dict(color="red", size=12)
        )

    # Layout styling
    fig.update_layout(
        height=900,
        width=None,
        autosize=True,
        template='plotly_white',
        showlegend=True,
        title=dict(
            text="📊 Real-time Attendance Analytics Dashboard",
            x=0.5,
            font=dict(size=24, color='#2c3e50')
        ),
        font=dict(family="Arial, sans-serif", size=12),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=50, r=50, t=100, b=50)
    )

    # Axis labels
    fig.update_xaxes(title_text="College Name", row=1, col=1, tickangle=45, title_font=dict(size=12))
    fig.update_yaxes(title_text="Count", row=1, col=1)

    fig.update_xaxes(title_text="Date", row=2, col=1, tickangle=45, title_font=dict(size=12))
    fig.update_yaxes(title_text="Count", row=2, col=1)

    fig.update_xaxes(title_text="Hours", row=2, col=2)
    fig.update_yaxes(title_text="Frequency", row=2, col=2)

    # Streamlit-compatible rendering
    # st.plotly_chart(fig, use_container_width=True)

    # Save static image for fallback
    try:
        fig.write_image(CHART_FILE, format="png", width=1200, height=800, scale=2)
        logger.debug(f"Saved analytics chart to {CHART_FILE}")
    except Exception as e:
        logger.error(f"Failed to save chart: {str(e)}")
    
    return fig, college_stats, status_counts, daily_trends

# Admin dashboard with enhanced analytics
def admin_dashboard():
    st.title(f'📊 Admin Dashboard - {st.session_state.user_role.title()}')
    
    if st.button('🚪 Logout'):
        st.session_state.authenticated = False
        st.session_state.user_role = None
        logger.info(f"Admin {st.session_state.user_role} logged out")
        st.rerun()
    
    st.markdown('---')
    
    # Auto-refresh toggle
    col1, col2 = st.columns([3, 1])
    with col2:
        auto_refresh = st.checkbox('🔄 Auto-refresh (30s)', value=False)
        if auto_refresh:
            time.sleep(30)
            st.rerun()
    
    df = load_records()
    
    if df.empty or 'username' not in df.columns:
        st.info('📝 No attendance records found.')
        return
    
    stats = calculate_summary_stats(df)
    
    # Enhanced metrics display
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    with col1:
        st.metric('👥 Total Interns', stats['total_interns'])
    with col2:
        st.metric('🏫 Total Colleges', stats['total_colleges'])
    with col3:
        st.metric('📅 Active Days', stats['active_days'])
    with col4:
        st.metric('⏰ Avg Hours', f"{stats['avg_hours']}h")
    with col5:
        st.metric('🔢 Total Hours', f"{stats['total_hours']}h")
    with col6:
        st.metric('✅ Complete Sessions', stats['complete_sessions'])
    
    # Today's stats
    st.markdown('### 📅 Today\'s Statistics')
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric('🟢 Checked In Today', stats['checked_in_today'])
    with col2:
        st.metric('🔴 Checked Out Today', stats['checked_out_today'])
    with col3:
        pending = stats['checked_in_today'] - stats['checked_out_today']
        st.metric('⏳ Pending Check-out', max(0, pending))
    
    st.markdown('---')
    
    # # Enhanced Analytics Dashboard
    # st.header('📈 Interactive Analytics Dashboard')
    
    analytics_result = generate_enhanced_analytics(df)
    if analytics_result:
        fig, college_stats, status_counts, daily_trends = analytics_result
        st.plotly_chart(
            fig,
            use_container_width=True,
            config={
                'displayModeBar': True,
                'modeBarButtonsToAdd': ['downloadImage', 'zoom2d', 'pan2d', 'select2d', 'lasso2d', 'autoScale2d', 'resetScale2d'],
                'toImageButtonOptions': {
                    'format': 'png',
                    'filename': f"Attendance_Analytics_{datetime.now(IST).strftime('%Y%m%d_%H%M')}",
                    'height': 800,
                    'width': 1200,
                    'scale': 2
                },
                'displaylogo': False
            }
        )
    
    # # Fallback static image
    # if os.path.exists(CHART_FILE):
    #     st.image(CHART_FILE, caption='Static Attendance Analytics', use_container_width=True)
    
    # Tabs for detailed views
    tab1, tab2, tab3, tab4, tab5 = st.tabs(['📋 All Records', '🏫 College Stats', '📊 Daily Trends', '💾 Export Data', '👤 Interns'])
    
    with tab1:
        st.subheader('All Attendance Records')
        
        col1, col2, col3 = st.columns(3)
        with col1:
            college_filter = st.selectbox('Filter by College', ['All'] + sorted(df['college_name'].unique().tolist()))
        with col2:
            status_filter = st.selectbox('Filter by Status', ['All'] + sorted(df['status'].unique().tolist()))
        with col3:
            date_filter = st.selectbox('Filter by Date', ['All'] + sorted(df['date'].unique().tolist(), reverse=True))
        
        filtered_df = df.copy()
        if college_filter != 'All':
            filtered_df = filtered_df[filtered_df['college_name'] == college_filter]
        if status_filter != 'All':
            filtered_df = filtered_df[filtered_df['status'] == status_filter]
        if date_filter != 'All':
            filtered_df = filtered_df[filtered_df['date'] == date_filter]
        
        display_df = filtered_df.copy()
        for col in ['in_time', 'out_time']:
            display_df[col] = display_df[col].replace('', 'Not marked')
        
        st.dataframe(
            display_df.sort_values(by=['date', 'in_time'], ascending=[False, False]),
            use_container_width=True,
            column_config={
                'username': 'Username',
                'college_name': 'College',
                'date': 'Date',
                'in_time': 'Check-in Time',
                'out_time': 'Check-out Time',
                'total_hours': 'Hours',
                'status': 'Status',
                'device_id': 'Device ID'
            }
        )
    
    with tab2:
        st.subheader('College-wise Statistics')
        if analytics_result and not college_stats.empty:
            st.dataframe(
                college_stats,
                use_container_width=True,
                column_config={
                    'College Name': 'College',
                    'Unique Interns': 'Interns',
                    'Active Days': 'Days',
                    'Total Hours': 'Hours'
                }
            )
    
    with tab3:
        st.subheader('Daily Trends')
        if analytics_result and not daily_trends.empty:
            st.dataframe(
                daily_trends,
                use_container_width=True,
                column_config={
                    'Date': 'Date',
                    'Unique Interns': 'Interns',
                    'Check-ins': 'Check-ins',
                    'Total Hours': 'Hours'
                }
            )
    
    with tab4:
        st.subheader('Export Data')
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if not df.empty:
                csv = df.to_csv(index=False)
                st.download_button(
                    label='📥 Export All Records',
                    data=csv,
                    file_name=f"Attendance_Records_{datetime.now(IST).strftime('%Y%m%d_%H%M')}.csv",
                    mime='text/csv',
                    use_container_width=True
                )
        
        with col2:
            if analytics_result and not college_stats.empty:
                csv = college_stats.to_csv(index=False)
                st.download_button(
                    label='📊 Export College Stats',
                    data=csv,
                    file_name=f"College_Stats_{datetime.now(IST).strftime('%Y%m%d_%H%M')}.csv",
                    mime='text/csv',
                    use_container_width=True
                )
        
        with col3:
            if analytics_result and not daily_trends.empty:
                csv = daily_trends.to_csv(index=False)
                st.download_button(
                    label='📈 Export Daily Trends',
                    data=csv,
                    file_name=f"Daily_Trends_{datetime.now(IST).strftime('%Y%m%d_%H%M')}.csv",
                    mime='text/csv',
                    use_container_width=True
                )
    
    with tab5:
        st.subheader('Manage Interns')
        interns_df = load_interns()
        if not interns_df.empty:
            st.dataframe(
                interns_df,
                use_container_width=True,
                column_config={
                    'username': 'Username',
                    'college_name': 'College'
                }
            )
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown('#### Add Intern')
            with st.form('add_intern_form'):
                new_username = st.text_input('Username')
                new_college = st.text_input('College Name')
                add_button = st.form_submit_button('Add')
                
                if add_button and new_username:
                    try:
                        databases.create_document(
                            database_id=APPWRITE_DATABASE_ID,
                            collection_id=APPWRITE_INTERNS_COLLECTION_ID,
                            document_id='unique()',
                            data={
                                'username': new_username,
                                'college_name': new_college
                            }
                        )
                        st.success(f"✅ Added {new_username}")
                        logger.info(f"Added intern: {new_username}")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error adding intern: {str(e)}")
                        logger.error(f"Error adding intern {new_username}: {str(e)}")
        
        with col2:
            st.markdown('#### Delete Intern')
            with st.form('delete_intern_form'):
                delete_username = st.selectbox('Username', interns_df['username'].tolist())
                delete_button = st.form_submit_button('Delete')
                
                if delete_button and delete_username:
                    try:
                        result = databases.list_documents(
                            database_id=APPWRITE_DATABASE_ID,
                            collection_id=APPWRITE_INTERNS_COLLECTION_ID,
                            queries=[Query.equal('username', delete_username)]
                        )
                        if result['total'] > 0:
                            document_id = result['documents'][0]['$id']
                            databases.delete_document(
                                database_id=APPWRITE_DATABASE_ID,
                                collection_id=APPWRITE_INTERNS_COLLECTION_ID,
                                document_id=document_id
                            )
                            st.success(f"✅ Deleted {delete_username}")
                            logger.info(f"Deleted intern: {delete_username}")
                            st.rerun()
                        else:
                            st.error(f"Intern {delete_username} not found")
                            logger.warning(f"Intern {delete_username} not found")
                    except Exception as e:
                        st.error(f"Error deleting intern: {str(e)}")
                        logger.error(f"Error deleting intern {delete_username}: {str(e)}")
        
        with col3:
            st.markdown('#### Upload Interns CSV')
            with st.form('upload_interns_form'):
                uploaded_file = st.file_uploader("Choose CSV", type="csv")
                upload_button = st.form_submit_button('Upload')
                
                if upload_button and uploaded_file:
                    count = import_interns_csv(uploaded_file)
                    if count > 0:
                        st.success(f"✅ Imported {count} interns")
                        logger.info(f"Imported {count} interns via CSV")
                        st.rerun()
                    elif count == 0:
                        st.warning("No new interns imported")
                        logger.info("No new interns imported from CSV")

# Load interns for admin view
def load_interns():
    logger.debug("Loading interns")
    try:
        result = databases.list_documents(
            database_id=APPWRITE_DATABASE_ID,
            collection_id=APPWRITE_INTERNS_COLLECTION_ID,
            queries=[Query.limit(1000)]
        )
        logger.debug(f"Loaded {result['total']} interns")
        return pd.DataFrame([{
            'username': r.get('username', ''),
            'college_name': r.get('college_name', '')
        } for r in result['documents']])
    except Exception as e:
        st.error(f"Error loading interns: {str(e)}")
        logger.error(f"Error loading interns: {str(e)}", exc_info=True)
        return pd.DataFrame(columns=['username', 'college_name'])

# Intern interface
def intern_interface():
    st.title('📝 Summer of AI Internship Attendance')
    st.markdown('**Event**: Summer of AI Internship, Swecha Office, Gachibowli, Hyderabad')
    st.markdown('---')
    
    st.header('⚡ Quick Attendance')
    
    username = st.text_input(
        '👤 Code.Swecha.org Username',
        value=st.session_state.username,
        placeholder='Enter your username'
    )
    if username != st.session_state.username:
        st.session_state.username = username
        logger.debug(f"Updated session username: {username}")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button('🟢 MARK IN', use_container_width=True, type='primary'):
            if username.strip():
                college_name = verify_username(username.strip())
                if college_name:
                    time_info = get_current_time_info()
                    logger.debug(f"Mark IN time info: {time_info}")
                    if save_record(username.strip(), college_name, 'In'):
                        st.success(f"✅ Welcome {username}! Marked IN at {datetime.now(IST).strftime('%H:%M:%S')}")
                        # logger.info(f"Marked IN: {username}")
                        st.balloons()
            else:
                st.error('❌ Please enter a username.')
                logger.warning("IN attempt with empty username")
    
    with col2:
        if st.button('🔴 MARK OUT', use_container_width=True):
            if username.strip():
                college_name = verify_username(username.strip())
                if college_name:
                    time_info = get_current_time_info()
                    logger.debug(f"Mark OUT time info: {time_info}")
                    if save_record(username.strip(), college_name, 'Out'):
                        st.success(f"✅ Goodbye {username}! Marked OUT at {datetime.now(IST).strftime('%H:%M:%S')}")
                        logger.info(f"Marked OUT: {username}")
            else:
                st.error('❌ Please enter a username.')
                logger.warning("OUT attempt with empty username")
    
    if username.strip():
        df = load_records()
        if df.empty or 'username' not in df.columns:
            st.info("📝 No attendance records found.")
            logger.debug(f"No records for {username}")
            return
        
        # Filter records for the current user
        user_records = df[df['username'] == username.strip()]
        
        # Today's status
        today = datetime.now(IST).strftime("%Y-%m-%d")
        user_today = user_records[user_records['date'] == today]
        
        if not user_today.empty:
            record = user_today.iloc[0]
            st.markdown('### 📅 Your Today\'s Status')
            
            col1, col2, col3 = st.columns(3)
            with col1:
                in_status = '✅' if record['in_time'] else '❌'
                st.info(f"🟢 **IN**: {in_status} {record.get('in_time', 'Not marked')}")
            with col2:
                out_status = '✅' if record['out_time'] else '❌'
                st.info(f"🔴 **OUT**: {out_status} {record.get('out_time', 'Not marked')}")
            with col3:
                hours = record['total_hours'] if record['total_hours'] else 'Incomplete'
                st.info(f"⏰ **Hours**: {hours}")
            logger.debug(f"Displayed today's status for {username}")
        
        # Attendance Summary
        st.markdown('---')
        st.header('📊 Your Attendance Summary')
        if user_records.empty:
            st.info("No attendance records found for you.")
        else:
            # Calculate user-specific summary stats
            complete_records = user_records[(user_records['in_time'] != '') & (user_records['out_time'] != '')]
            stats = {
                'total_days': user_records['date'].nunique(),
                'complete_sessions': len(complete_records),
                'total_hours': round(pd.to_numeric(complete_records['total_hours'], errors='coerce').sum(), 2),
                'avg_hours': round(pd.to_numeric(complete_records['total_hours'], errors='coerce').mean(), 2) if not complete_records.empty else 0.0
            }
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric('📅 Total Days Attended', stats['total_days'])
            with col2:
                st.metric('✅ Complete Sessions', stats['complete_sessions'])
            with col3:
                st.metric('⏰ Total Hours', f"{stats['total_hours']}h")
            with col4:
                st.metric('📈 Avg Hours per Session', f"{stats['avg_hours']}h")
            logger.debug(f"Displayed attendance summary for {username}: {stats}")
        
        # Past Attendance Records
        st.markdown('---')
        st.header('📋 Your Past Attendance Records')
        if user_records.empty:
            st.info("No past attendance records found.")
        else:
            display_df = user_records.copy()
            for col in ['in_time', 'out_time']:
                display_df[col] = display_df[col].replace('', 'Not marked')
            
            st.dataframe(
                display_df.sort_values(by='date', ascending=False),
                use_container_width=True,
                column_config={
                    'username': 'Username',
                    'college_name': 'College',
                    'date': 'Date',
                    'in_time': 'Check-in Time',
                    'out_time': 'Check-out Time',
                    'total_hours': 'Hours',
                    'status': 'Status',
                    'device_id': 'Device ID'
                }
            )
            logger.debug(f"Displayed {len(user_records)} past records for {username}")
            
            # Export option
            csv = display_df.to_csv(index=False)
            st.download_button(
                label='📥 Export Your Records',
                data=csv,
                file_name=f"{username}_Attendance_{datetime.now(IST).strftime('%Y%m%d_%H%M')}.csv",
                mime='text/csv',
                use_container_width=True
            )
     
# Main application
def main():
    if DEBUG_MODE:
        with st.sidebar:
            st.markdown('### 🔧 Configuration Status')
            st.text(f"Admin Users: {len(ADMIN_CREDENTIALS)} configured")
            st.text(f"Debug Mode: {'ON' if DEBUG_MODE else 'OFF'}")
            st.text(f"Appwrite: {'Connected' if APPWRITE_PROJECT_ID else 'Not configured'}")
            time_info = get_current_time_info()
            st.text(f"Local Time: {time_info['local_time']}")
            st.text(f"IST Time: {time_info['ist_time']}")
            st.text(f"System TZ: {time_info['system_tz']}")
            st.text(f"Device ID: {st.session_state.device_fingerprint}")
    
    st.sidebar.title('🏢 Summer of AI Tracker')
    
    if not st.session_state.authenticated:
        mode = st.sidebar.radio('Select Mode', ['👤 Intern Attendance', '🔐 Admin Login'])
        
        if mode == '👤 Intern Attendance':
            intern_interface()
        else:
            admin_login()
    else:
        st.sidebar.success(f"Logged in as: {st.session_state.user_role.title()}")
        admin_dashboard()

if __name__ == '__main__':
    logger.debug("Starting application")
    time_info = get_current_time_info()
    logger.debug(f"Startup time info: {time_info}")
    if platform.system() == 'Emscripten':
        import asyncio
        async def async_main():
            main()
        asyncio.ensure_future(async_main())
    else:
        main()
