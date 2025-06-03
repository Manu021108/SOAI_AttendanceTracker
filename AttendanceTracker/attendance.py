import streamlit as st
import pandas as pd
from datetime import datetime
import os
import matplotlib.pyplot as plt
import seaborn as sns
from appwrite.client import Client
from appwrite.services.databases import Databases
from appwrite.query import Query
from dotenv import load_dotenv
import platform
import pytz
import logging
import time

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
    st.error(f"❌ Missing Appwrite configuration: {', '.join(missing)}")
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

# Timezone for IST
IST = pytz.timezone('Asia/Kolkata')

# Debug timezone
def get_current_time_info():
    try:
        local_time = datetime.datetime.now()
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
                'total_hours', 'status'
            ])
        df = pd.DataFrame([{
            'username': r.get('username', ''),
            'college_name': r.get('college_name', ''),
            'date': r.get('date', ''),
            'in_time': r.get('in_time', ''),
            'out_time': r.get('out_time', ''),
            'total_hours': float(r.get('total_hours', 0.0)),
            'status': r.get('status', '')
        } for r in result['documents']])
        # Ensure all columns exist
        required_columns = ['username', 'college_name', 'date', 'in_time', 'out_time', 
                           'total_hours', 'status']
        for col in required_columns:
            if col not in df.columns:
                df[col] = '' if col != 'total_hours' else 0.0
        return df
    except Exception as e:
        st.error(f"Error loading records: {str(e)}")
        logger.error(f"Error loading records: {str(e)}", exc_info=True)
        return pd.DataFrame(columns=[
            'username', 'college_name', 'date', 'in_time', 'out_time',
            'total_hours', 'status'
        ])

# Save attendance record to Appwrite
def save_record(username, college_name, action):
    logger.debug(f"Saving record: username={username}, action={action}")
    time_info = get_current_time_info()
    logger.debug(f"Timezone info: {time_info}")
    today = datetime.now(IST).strftime("%Y-%m-%d")
    current_time = datetime.now(IST).strftime("%H:%M:%S")
    logger.debug(f"Recording time: {today} {current_time} IST")
    
    try:
        result = databases.list_documents(
            database_id=APPWRITE_DATABASE_ID,
            collection_id=APPWRITE_ATTENDANCE_COLLECTION_ID,
            queries=[
                Query.equal('username', username),
                Query.equal('date', today)
            ]
        )
        existing_records = result['documents']
        logger.debug(f"Found {len(existing_records)} existing records for {username} today")
        
        if action == "In":
            if existing_records:
                existing_record = existing_records[0]
                if existing_record.get('in_time'):
                    if existing_record.get('in_time'):
                    st.warning("⚠️ You have already marked IN today.")
                    logger.warning(f"Duplicate IN blocked for {username}")
                    return False

                databases.update_document(
                    database_id=APPWRITE_DATABASE_ID,
                    collection_id=APPWRITE_ATTENDANCE_COLLECTION_ID,
                    document_id=existing_record['$id'],
                    data={
                        'in_time': current_time,
                        'in_ip': '',
                        'status': 'Checked In',
                        'out_time': '',
                        'out_ip': '',
                        'total_hours': 0.0
                    }
                )
                logger.debug(f"Updated IN record for {username}")
            else:
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
                        'in_ip': '',
                        'out_ip': '',
                        'status': 'Checked In'
                    }
                )
                logger.debug(f"Created IN record for {username}")
        
        elif action == "Out":
            if not existing_records or not existing_records[0].get('in_time'):
                st.error("❌ Cannot mark OUT without marking IN first today.")
                logger.warning(f"OUT blocked: no IN record for {username}")
                return False
            
            existing_record = existing_records[0]
            if existing_record.get('out_time'):
                st.warning("⚠️ You have already marked OUT today.")
                logger.warning(f"Duplicate OUT blocked for {username}")
                return False

            
            try:
                in_time_dt = datetime.strptime(existing_record['in_time'], "%H:%M:%S")
                out_time_dt = datetime.strptime(current_time, "%H:%M:%S")
                total_seconds = (out_time_dt - in_time_dt).total_seconds()
                total_hours = float(round(total_seconds / 3600, 2))
            except Exception as e:
                st.warning(f"Error calculating hours: {str(e)}")
                logger.warning(f"Error calculating hours for {username}: {str(e)}")
                total_hours = 0.0
            
            databases.update_document(
                database_id=APPWRITE_DATABASE_ID,
                collection_id=APPWRITE_ATTENDANCE_COLLECTION_ID,
                document_id=existing_record['$id'],
                data={
                    'out_time': current_time,
                    'out_ip': '',
                    'total_hours': total_hours,
                    'status': 'Checked Out'
                }
            )
            logger.debug(f"Updated OUT record for {username}")
        return True
    except Exception as e:
        st.error(f"Error saving record: {str(e)}")
        logger.error(f"Error saving record for {username}: {str(e)}", exc_info=True)
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
            'complete_sessions': 0
        }
    
    complete_records = df[(df['in_time'] != '') & (df['out_time'] != '') & (df['total_hours'] != '')]
    
    stats = {
        'total_interns': df['username'].nunique(),
        'total_colleges': df['college_name'].nunique() if 'college_name' in df.columns else 0,
        'total_records': len(df),
        'active_days': df['date'].nunique() if 'date' in df.columns else 0,
        'avg_hours': 0,
        'total_hours': 0,
        'complete_sessions': len(complete_records)
    }
    
    if not complete_records.empty:
        hours_data = pd.to_numeric(complete_records['total_hours'], errors='coerce').dropna()
        if not hours_data.empty:
            stats['avg_hours'] = round(hours_data.mean(), 2)
            stats['total_hours'] = round(hours_data.sum(), 2)
    
    logger.debug(f"Summary stats: {stats}")
    return stats

# Generate analytics
def generate_analytics(df):
    logger.debug("Generating analytics")
    if df.empty or 'username' not in df.columns:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    
    college_stats = df.groupby('college_name').agg({
        'username': 'nunique',
        'date': 'nunique',
        'total_hours': lambda x: pd.to_numeric(x, errors='coerce').sum()
    }).reset_index()
    college_stats.columns = ['College Name', 'Unique Interns', 'Active Days', 'Total Hours']
    college_stats['Total Hours'] = college_stats['Total Hours'].round(2)
    
    status_counts = df['status'].value_counts().reset_index()
    status_counts.columns = ['Status', 'Count']
    
    daily_trends = df.groupby('date').agg({
        'username': 'nunique',
        'status': lambda x: (x == 'Checked In').sum(),
        'total_hours': lambda x: pd.to_numeric(x, errors='coerce').sum()
    }).reset_index()
    daily_trends.columns = ['Date', 'Unique Interns', 'Check-ins', 'Total Hours']
    daily_trends['Total Hours'] = daily_trends['Total Hours'].round(2)
    
    plt.style.use('default')
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    if not college_stats.empty:
        bars = ax1.bar(college_stats['College Name'], college_stats['Unique Interns'], 
                       color='skyblue', edgecolor='navy', alpha=0.7)
        ax1.set_xlabel('College Name')
        ax1.set_ylabel('Unique Interns')
        ax1.set_title('Unique Interns by College')
        ax1.tick_params(axis='x', rotation=45)
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                     f'{int(height)}', ha='center', va='bottom')
    
    if not status_counts.empty:
        colors = ['lightgreen' if 'In' in status else 'lightcoral' for status in status_counts['Status']]
        ax2.pie(status_counts['Count'], labels=status_counts['Status'], 
                autopct='%1.1f%%', startangle=90, colors=colors)
        ax2.set_title('Status Distribution')
    
    if not daily_trends.empty:
        ax3.plot(daily_trends['Date'], daily_trends['Unique Interns'], 
                 marker='o', label='Unique Interns', linewidth=2, color='blue')
        ax3.plot(daily_trends['Date'], daily_trends['Check-ins'], 
                 marker='s', label='Check-ins', linewidth=2, color='green')
        ax3.set_xlabel('Date')
        ax3.set_ylabel('Count')
        ax3.set_title('Daily Attendance Trends')
        ax3.legend()
        ax3.tick_params(axis='x', rotation=45)
    
    hours_data = pd.to_numeric(df['total_hours'], errors='coerce').dropna()
    if not hours_data.empty:
        ax4.hist(hours_data, bins=max(10, len(hours_data)//5), 
                 color='gold', alpha=0.7, edgecolor='orange')
        ax4.set_xlabel('Hours Spent')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Distribution of Session Hours')
        ax4.axvline(hours_data.mean(), color='red', linestyle='--', 
                    label=f'Mean: {hours_data.mean():.2f}h')
        ax4.legend()
    
    plt.tight_layout()
    plt.savefig(CHART_FILE, dpi=300, bbox_inches='tight')
    plt.close()
    logger.debug(f"Saved analytics chart to {CHART_FILE}")
    
    return college_stats, status_counts, daily_trends

# Admin login function
def admin_login():
    st.title('🔐 Admin Login')
    st.markdown('---')
    
    with st.form('login_form'):
        username = st.text_input('Username')
        password = st.text_input('Password', type='password')
        login_button = st.form_submit_button('Login')
        
        if login_button:
            if username in ADMIN_CREDENTIALS and ADMIN_CREDENTIALS[username] == password:
                st.session_state.authenticated = True
                st.session_state.user_role = username
                st.success(f'✅ Welcome, {username.title()}!')
                logger.info(f"Admin {username} logged in")
                st.rerun()
            else:
                st.error('❌ Invalid credentials.')
                logger.warning(f"Failed login attempt for {username}")

# Admin dashboard
def admin_dashboard():
    st.title(f'📊 Admin Dashboard - {st.session_state.user_role.title()}')
    
    if st.button('🚪 Logout'):
        st.session_state.authenticated = False
        st.session_state.user_role = None
        logger.info(f"Admin {st.session_state.user_role} logged out")
        st.rerun()
    
    st.markdown('---')
    
    df = load_records()
    
    if df.empty or 'username' not in df.columns:
        st.info('📝 No attendance records found.')
        return
    
    stats = calculate_summary_stats(df)
    
    col1, col2, col3, col4, col5 = st.columns(5)
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
    
    st.markdown('---')
    
    st.header('📈 Analytics Dashboard')
    college_stats, status_counts, daily_trends = generate_analytics(df)
    
    if os.path.exists(CHART_FILE):
        st.image(CHART_FILE, caption='Comprehensive Attendance Analytics', use_container_width=True)
    
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
        
        st.dataframe(display_df.sort_values(['date', 'in_time'], ascending=[False, False]), 
                     use_container_width=True)
    
    with tab2:
        st.subheader('College-wise Statistics')
        if not college_stats.empty:
            st.dataframe(college_stats, use_container_width=True)
    
    with tab3:
        st.subheader('Daily Trends')
        if not daily_trends.empty:
            st.dataframe(daily_trends, use_container_width=True)
    
    with tab4:
        st.subheader('Export Data')
        col1, col2 = st.columns(2)
        
        with col1:
            if not df.empty:
                csv = df.to_csv(index=False)
                st.download_button(
                    label='📥 Export All Records',
                    data=csv,
                    file_name=f"Attendance_Records_{datetime.now(IST).strftime('%Y%m%d_%H%M')}.csv",
                    mime='text/csv'
                )
        
        with col2:
            if not college_stats.empty:
                csv = college_stats.to_csv(index=False)
                st.download_button(
                    label='📊 Export College Stats',
                    data=csv,
                    file_name=f"College_Statistics_{datetime.now(IST).strftime('%Y%m%d_%H%M')}.csv",
                    mime='text/csv'
                )
    
    with tab5:
        st.subheader('Manage Interns')
        interns_df = load_interns()
        st.dataframe(interns_df, use_container_width=True)
        
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
    
    username = st.text_input('👤 Code.Swecha.org Username', 
                             value=st.session_state.username,
                             placeholder='Enter your username')
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
                        logger.info(f"Marked IN: {username}")
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
        today = datetime.now(IST).strftime("%Y-%m-%d")
        if df.empty or 'username' not in df.columns:
            st.info("📝 No attendance records found for today.")
            logger.debug(f"No records for {username} today")
            return
        
        user_today = df[(df['username'] == username.strip()) & (df['date'] == today)]
        
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