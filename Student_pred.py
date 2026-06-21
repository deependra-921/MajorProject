import streamlit as st
import matplotlib.pyplot as plt
import mysql.connector

import base64
import joblib
from huggingface_hub import InferenceClient

# ---------------- ML MODEL ----------------
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Student Performance System", layout="wide")

# ---------------- HF CLIENT (key pulled from st.secrets, not hardcoded) ----------------
client = InferenceClient(
    api_key=st.secrets["HF_API_KEY"]
)

# ---------------- BACKGROUND ----------------
def get_base64(file):
    with open(file, "rb") as f:
        return base64.b64encode(f.read()).decode()

try:
    img = get_base64("bg.jpg")
except FileNotFoundError:
    st.error("⚠️ Background image not found")
    img = ""
except Exception as e:
    st.error(f"⚠️ Error loading background image: {e}")
    img = ""

st.markdown(f"""
<style>
.stApp {{
    background: linear-gradient(rgba(0,0,0,0.5), rgba(0,0,0,0.5)),
                url("data:image/jpg;base64,{img}");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    background-attachment: fixed;
}}
</style>
""", unsafe_allow_html=True)

# ---------------- DATABASE ----------------

def get_connection():
    try:
        conn = mysql.connector.connect(
            host=st.secrets["DB_HOST"],
            user=st.secrets["DB_USER"],
            password=st.secrets["DB_PASSWORD"],
            database=st.secrets["DB_NAME"],
            port=int(st.secrets.get("DB_PORT", 3306))
        )
        return conn
    except mysql.connector.Error as e:
        st.error(f"❌ Database Connection Error: {e}")
        return None

# ---------------- CHATBOT KNOWLEDGE ----------------

# ---------------- PROJECT KNOWLEDGE BASE ----------------

project_info = """
You are the official AI assistant of the Student Performance System project.

PROJECT OVERVIEW:
This project is an AI-powered Student Performance Analysis and Prediction System developed using:
- Python
- Streamlit
- Machine Learning
- MySQL
- NLTK
- Gemini AI
- Pandas
- Matplotlib
- Scikit-learn

==================================================
LOGIN & SIGNUP SYSTEM
==================================================

LOGIN:
Students log in using:
- Name
- Class/Course
- Roll Number
- Year

The login system checks data from MySQL database.

SIGNUP:
Students can:
- Create account
- Add multiple subjects
- Save subjects into database

DATABASE TABLES:
1. student
2. subjects

==================================================
MARKS ANALYSIS SYSTEM
==================================================

FEATURES:
1. Calculate Average
2. Find Percentage
3. Pie Chart Visualization
4. Student Comparison
5. Student Overview

AVERAGE FORMULA:
average = sum(marks) / total_subjects

PERCENTAGE FORMULA:
percentage = (obtained_marks / total_marks) * 100

==================================================
STUDENT ANALYSIS
==================================================

Student Analysis compares:
- Previous marks
- Current marks

The system categorizes:
- Good Subjects (80+)
- Average Subjects (50–79)
- Weak Subjects (Below 50)

Green bars:
Improved performance

Red bars:
Performance decreased

==================================================
OVERVIEW SYSTEM
==================================================

Grades:
90+ = A+
75+ = A
60+ = B
33+ = C
Below 33 = D

Performance:
A+ = Excellent
A = Very Good
B = Good
C = Average
D = Poor

==================================================
AI SUGGESTIONS SYSTEM
==================================================

The system gives suggestions based on:
- Percentage
- Weak subjects
- Marks

Rules:
- Below 50% → increase study time
- Weak subjects → revise regularly
- Above 75% → try advanced problems

==================================================
RISK LEVEL SYSTEM
==================================================

Risk Categories:
Below 40 → High Risk
40–60 → Moderate Risk
Above 60 → Low Risk

==================================================
SMART STUDY PLAN
==================================================

The study planner:
- Detects weak subjects
- Detects strong subjects
- Suggests study hours
- Suggests learning strategy

Examples:
Math:
- numerical practice

Programming:
- coding practice

Science:
- concepts + formulas

Languages:
- grammar + writing

==================================================
GOAL SETTER
==================================================

The goal setter checks:
- Current percentage
- Target percentage
- Attendance
- Participation

Then suggests:
- Study strategy
- Attendance improvement
- Participation improvement

==================================================
MACHINE LEARNING PREDICTION SYSTEM
==================================================

Algorithm Used:
Logistic Regression

Purpose:
Predict whether student may PASS or FAIL.

INPUT FEATURES:
1. Weekly Study Hours
2. Attendance Percentage
3. Class Participation
4. Study Attendance Combo
5. Engagement Score

FEATURE ENGINEERING:

study_attendance_combo =
study_hours * attendance

engagement_score =
attendance + participation * 10

TARGET LOGIC:
Student predicted PASS if:
- attendance >=85 and participation >=4
OR
- study_hours >=5 and attendance >=60
OR
- participation ==5 and attendance >=50

Otherwise FAIL.

==================================================
WHY STUDENT MAY FAIL
==================================================

Possible reasons:
- Low attendance
- Low study hours
- Low participation
- Weak engagement score

==================================================
WHY STUDENT MAY PASS
==================================================

Possible reasons:
- High attendance
- Good participation
- Good study hours
- Strong engagement score

==================================================
MODEL TRAINING
==================================================

Steps:
1. Load dataset
2. Feature engineering
3. Train-test split
4. Standard scaling
5. Train Logistic Regression
6. Predict PASS/FAIL

==================================================
TOOLS & LIBRARIES USED
==================================================

Frontend:
- Streamlit

Visualization:
- Matplotlib

Database:
- MySQL

Machine Learning:
- Scikit-learn

AI Chatbot:
- Gemini API

NLP:
- NLTK

Data Handling:
- Pandas

==================================================
CHATBOT RESPONSIBILITIES
==================================================

The chatbot must:
- Explain any project feature
- Explain calculations
- Explain prediction logic
- Explain graphs
- Explain login/signup
- Explain ML concepts
- Explain database usage
- Explain why prediction failed
- Explain tools used
- Answer viva questions
- Answer technical questions
- Answer tricky project questions

The chatbot should answer:
- professionally
- accurately
- clearly
- with examples
- in detail
"""
# ---------------- SESSION ----------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "user" not in st.session_state:
    st.session_state.user = {}

if "page" not in st.session_state:
    st.session_state.page = "login"

if "app_page" not in st.session_state:
    st.session_state.app_page = "main"

# ---------------- LOGIN / SIGNUP ----------------
if not st.session_state.logged_in:

    if st.session_state.page == "login":

        st.title("🔐 Student Login")

        name = st.text_input("Name")
        clas = st.text_input("Class or course")
        roll = st.text_input("Roll")
        yop = st.text_input("Year")

        if st.button("Login"):
            try:
                conn = get_connection()

                if conn is None:
                    st.stop()

                cursor = conn.cursor()

                query = """
                SELECT * FROM student 
                WHERE name=%s AND class_course=%s AND roll_no=%s AND yop=%s
                """

                cursor.execute(query, (name, clas, roll, yop))
                result = cursor.fetchone()

                if result:
                    st.success("✅ Login Successful")
                    st.session_state.logged_in = True
                    st.session_state.user = {
                        "name": name,
                        "roll": roll,
                        "class": clas,
                        "yop": yop
                    }
                    st.rerun()
                else:
                    st.error("❌ Invalid Details")

            except Exception as e:
                st.error(f"Error: {e}")

            finally:
                if conn and conn.is_connected():
                    conn.close()
        if st.button("Go to Signup"):
            st.session_state.page = "signup"
            st.rerun()

    elif st.session_state.page == "signup":

        st.title("📝 Sign Up")

        name = st.text_input("Name")
        roll = st.text_input("Roll")
        yop = st.text_input("Year")
        clas = st.radio("Choose class or course :", ["class", "course"])
        inp = st.text_input(f"Enter your {clas}")

        st.title("📚 Add Subjects")

        if "sub" not in st.session_state:
            st.session_state.sub = []

        s = st.text_input("Enter Subject")

        if st.button("Enter"):
            if s and s not in st.session_state.sub:
                st.session_state.sub.append(s)
                st.rerun()

        if st.session_state.sub:
            st.write("### Your Subjects:")
            for i, subject in enumerate(st.session_state.sub, 1):
                st.write(f"{i}. {subject}")

        if st.button("Submit"):
            st.success("Subjects saved successfully!")

        # ✅ UPDATED SIGNUP (SAVE SUBJECTS IN DB)
        if st.button("Sign Up"):
            try:
                conn = get_connection()

                if conn is None:
                    st.stop()

                cursor = conn.cursor()

                # student table
                cursor.execute(
                    "INSERT INTO student (name, class_course, roll_no, yop) VALUES (%s,%s,%s,%s)",
                    (name, inp, roll, yop)
                )

                # ✅ insert subjects
                for sub in st.session_state.sub:
                    cursor.execute(
                        "INSERT INTO subjects (roll_no, subject) VALUES (%s,%s)",
                        (roll, sub)
                    )


                conn.commit()

                st.success("✅ Signup Successful")
                st.session_state.page = "login"
                st.session_state.sub = []
                st.rerun()

            except mysql.connector.IntegrityError:
                st.error("❌ User already exists")

            finally:
                conn.close()

        if st.button("Back to Login"):
            st.session_state.page = "login"
            st.rerun()

# ---------------- MAIN APP ----------------
else:

    st.sidebar.title("👤 Student Info")
    st.sidebar.write(f"**Name:** {st.session_state.user['name']}")
    st.sidebar.write(f"**Roll:** {st.session_state.user['roll']}")
    st.sidebar.write(f"**Year:** {st.session_state.user['yop']}")

    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.rerun()

    if st.sidebar.button("🎯 Prediction"):
        st.session_state.app_page = "prediction"
        st.rerun()

    if st.sidebar.button("🏠 Back to Home"):
        st.session_state.app_page = "main"
        st.rerun()

    if st.sidebar.button("🤖 AI Chatbot"):
        st.session_state.app_page = "chatbot"
        st.rerun()
    # ---------------- MAIN SCREEN ----------------
    if st.session_state.app_page == "main":

        st.title("📊 Student Performance System")

        # ✅ LOAD SUBJECTS FROM DATABASE
        try:
            conn = get_connection()

            if conn is None:
                st.stop()

            cursor = conn.cursor()

            cursor.execute(
                "SELECT subject FROM subjects WHERE roll_no=%s",
                (st.session_state.user["roll"],)
            )

            subjects_data = cursor.fetchall()

            conn.close()

        except mysql.connector.Error as e:
            st.error(f"❌ Error loading subjects: {e}")
            subjects_data = []

        except Exception as e:
            st.error(f"⚠️ Unexpected Error: {e}")
            subjects_data = []

        subjects = [s[0] for s in subjects_data]

        marks = []

        st.header("Enter Marks")

        cols = st.columns(len(subjects) if subjects else 1)

        for i, sub in enumerate(subjects):
            with cols[i]:
                m = st.number_input(sub, 0, 100, key=sub)
                marks.append(m)

        if st.button("Calculate Average") and marks:
            avg = sum(marks)/len(marks)
            fig, ax = plt.subplots()
            ax.plot([0, avg], marker='o')
            ax.set_title(f"Average = {avg:.2f}")
            st.pyplot(fig)

        if st.button("Find Percentage") and marks:
            per = (sum(marks)/(len(marks)*100))*100
            fig, ax = plt.subplots()
            ax.plot([0, per], marker='o')
            ax.set_title(f"Percentage = {per:.2f}%")
            st.pyplot(fig)

        if st.button("Marks Chart") and marks:
            fig, ax = plt.subplots()
            ax.pie(marks, labels=subjects, autopct='%1.1f%%')
            st.pyplot(fig)

    # ---------------- COMPARISON FORM ----------------
        # ---------------- DEFINE SUBJECTS ----------------


        # ---------------- COMPARISON FORM ----------------
        st.header("📈 Student Analysis")

        with st.form("analysis_form"):

            prev_marks = []
            curr_marks = []

            # ✅ create dynamic columns
            cols = st.columns(len(subjects))

            for i, sub in enumerate(subjects):
                with cols[i]:
                    prev = st.number_input(f"{sub} Prev", 0, 100, key=sub + "_p")
                    curr = st.number_input(f"{sub} Curr", 0, 100, key=sub + "_c")

                    prev_marks.append(prev)
                    curr_marks.append(curr)

            # ✅ buttons
            col_btn1, col_btn3 = st.columns(2)

            with col_btn1:
                compare_btn = st.form_submit_button("📊 Compare")



            with col_btn3:
                overview_btn = st.form_submit_button("📋 Overview")

        # ---------------- COMPARE ----------------
        if compare_btn:
            colors = ["green" if c >= p else "red" for p, c in zip(prev_marks, curr_marks)]

            fig, axes = plt.subplots(1, 2, figsize=(12, 5))

            axes[0].bar(subjects, prev_marks)
            axes[0].set_title("Previous")

            axes[1].bar(subjects, curr_marks, color=colors)
            axes[1].set_title("Current")

            st.pyplot(fig)

            # -------- SUMMARY --------
            good = [s for s, m in zip(subjects, curr_marks) if m >= 80]
            avg = [s for s, m in zip(subjects, curr_marks) if 50 <= m < 80]
            low = [s for s, m in zip(subjects, curr_marks) if m < 50]

            st.subheader("📊 Summary")

            st.write("✅ Good Subjects:", ", ".join(good) if good else "None")
            st.write("⚖️ Average Subjects:", ", ".join(avg) if avg else "None")
            st.write("⚠️ Need Improvement:", ", ".join(low) if low else "None")





        # ---------------- OVERVIEW ----------------
        if overview_btn:

            total = sum(curr_marks)
            percentage = (total / (len(subjects) * 100)) * 100  # ✅ dynamic fix

            if percentage >= 90:
                grade, perf = "A+", "Excellent"
            elif percentage >= 75:
                grade, perf = "A", "Very Good"
            elif percentage >= 60:
                grade, perf = "B", "Good"
            elif percentage >= 33:
                grade, perf = "C", "Average"
            else:
                grade, perf = "D", "Poor"

            result = "Passed" if percentage >= 33 else "Failed"

            st.subheader("📋 Overview")

            st.table({
                "Name": [st.session_state.user["name"]],
                "Class": [st.session_state.user["class"]],
                "Roll": [st.session_state.user["roll"]],
                "Year": [st.session_state.user["yop"]],
                "Percentage": [f"{percentage:.2f}%"],
                "Grade": [grade],
                "Performance": [perf],
                "Result": [result]
            })
        # ================= AI FEATURES START =================

        # 🤖 AI Suggestions
        # ================= AI ANALYSIS DASHBOARD =================

        st.header("🤖 AI Analysis Dashboard")

        # ---------------- AI Suggestions ----------------
        st.subheader("🤖 AI Suggestions")

        ai_percentage = st.number_input("Enter Percentage", 0.0, 100.0, key="sug_per")
        ai_marks = st.text_input("Enter Marks (comma separated)", key="sug_marks")

        if st.button("Get Suggestions 🚀"):

            suggestions = []

            try:
                marks_list = [int(x) for x in ai_marks.split(",")] if ai_marks else []

            except ValueError:
                st.error("❌ Enter valid marks separated by commas")
                marks_list = []

            if ai_percentage < 50:
                suggestions.append("Increase study hours to at least 5–6 hrs/week")

            if ai_percentage < 60:
                suggestions.append("Focus more on weak subjects")

            if any(m < 50 for m in marks_list):
                suggestions.append("Revise low-scoring subjects regularly")

            if ai_percentage > 75:
                suggestions.append("Great job! Try advanced problems")

            if suggestions:
                for s in suggestions:
                    st.write("👉", s)
            else:
                st.success("✅ You are on the right track!")



        # ---------------- Risk Level ----------------
        st.subheader("⚠️ Risk Level")

        risk_percentage = st.number_input("Enter Percentage for Risk", 0.0, 100.0, key="risk")

        if st.button("Check Risk"):

            if risk_percentage < 40:
                st.error("High Risk of Failure")
            elif risk_percentage < 60:
                st.warning("Moderate Risk")
            else:
                st.success("Low Risk")

        # ---------------- Study Plan ----------------
        st.subheader("📅 Personalized Study Plan (AI Dynamic)")

        subjects_input = st.text_input("Subjects (comma separated)", key="subj")
        marks_input = st.text_input("Marks (comma separated)", key="marks")

        if st.button("Generate Smart Plan 🚀"):

            try:
                subjects_list = [s.strip() for s in subjects_input.split(",")] if subjects_input else []
                marks_list = [int(x) for x in marks_input.split(",")] if marks_input else []

                if len(subjects_list) != len(marks_list):
                    st.error("❌ Subjects and Marks count must match")
                else:
                    plan_data = []

                    for sub, mark in zip(subjects_list, marks_list):

                        sub_lower = sub.lower()

                        # ---------- LEVEL ----------
                        if mark < 50:
                            level = "Weak"
                            time = "2 hrs/day"
                        elif mark < 75:
                            level = "Average"
                            time = "1–1.5 hrs/day"
                        else:
                            level = "Strong"
                            time = "30–45 mins/day"

                        # ---------- SMART CATEGORY DETECTION ----------
                        if any(k in sub_lower for k in ["math", "stat", "account"]):
                            suggestion = "Practice problem-solving and numerical questions daily"

                        elif any(k in sub_lower for k in ["english", "hindi", "language"]):
                            suggestion = "Focus on writing, grammar, and reading comprehension"

                        elif any(k in sub_lower for k in ["science", "physics", "chemistry", "bio"]):
                            suggestion = "Understand concepts deeply and revise diagrams/formulas"

                        elif any(k in sub_lower for k in ["computer", "cs", "program", "ai"]):
                            suggestion = "Practice coding and strengthen core concepts"

                        elif any(k in sub_lower for k in ["history", "geo", "civics", "economics"]):
                            suggestion = "Focus on theory, revision, and short notes"

                        else:
                            # ✅ FULLY GENERIC FALLBACK
                            suggestion = "Revise concepts, make notes, and practice regularly"

                        # ---------- WEAK BOOST ----------
                        if mark < 50:
                            suggestion += " (High priority: daily focus required)"

                        # ---------- ADD TO TABLE ----------
                        plan_data.append({
                            "Subject": sub,
                            "Marks": mark,
                            "Level": level,
                            "Daily Time": time,
                            "AI Suggestion": suggestion
                        })

                    plan_df = pd.DataFrame(plan_data)

                    st.subheader("📊 AI Study Plan Table")
                    st.dataframe(plan_df, use_container_width=True)
            except ValueError:
                st.error("❌ Please enter valid numeric marks")

            except Exception as e:
                st.error(f"⚠️ Error generating study plan: {e}")

        # ---------------- Strength Graph ----------------
        st.subheader("📊 Strength Analysis")

        graph_subjects = st.text_input("Subjects for Graph", key="gsub")
        graph_marks = st.text_input("Marks for Graph", key="gmarks")

        if st.button("Show Graph"):

            try:

                subs = graph_subjects.split(",") if graph_subjects else []
                marks = [int(x) for x in graph_marks.split(",")] if graph_marks else []

                if len(subs) != len(marks):
                    st.error("❌ Subjects and marks count must match")

                elif subs and marks:

                    fig, ax = plt.subplots()
                    ax.bar(subs, marks)
                    ax.set_title("Subject Strength")

                    st.pyplot(fig)

            except ValueError:
                st.error("❌ Enter valid numeric marks")

            except Exception as e:
                st.error(f"⚠️ Graph Error: {e}")

        # ---------------- Goal Setter ----------------
        st.subheader("🎯 Goal Setter (AI Guided)")

        current_per = st.number_input("Current Percentage", 0.0, 100.0, key="goal_curr")
        target_per = st.number_input("Target Percentage", 0.0, 100.0, key="goal_target")

        attendance = st.number_input("Your Attendance %", 0, 100, key="goal_att")
        participation = st.number_input("Class Participation (1-5)", 1, 5, key="goal_part")

        if st.button("Check Goal 🚀"):

            gap = target_per - current_per

            if current_per >= target_per:
                st.success("🎉 Goal Achieved!")

            else:
                st.warning(f"You need {gap:.2f}% more to reach your goal")

                st.subheader("📊 Improvement Strategy")

                # -------- STUDY PLAN --------
                if gap > 20:
                    st.write("📌 Increase study time to 6–7 hrs/week")
                    st.write("📌 Focus on weak subjects daily")

                elif gap > 10:
                    st.write("📌 Study consistently 4–5 hrs/week")
                    st.write("📌 Practice previous year questions")

                else:
                    st.write("📌 Small improvement needed, focus on revision")

                # -------- ATTENDANCE --------
                if attendance < 60:
                    st.write("📌 Improve attendance to at least 75%")

                # -------- PARTICIPATION --------
                if participation < 3:
                    st.write("📌 Participate more in class (ask questions, interact)")

                # -------- MOTIVATION --------
                st.subheader("💡 Smart Tip")

                if gap > 15:
                    st.write("Consistency is key — follow a daily schedule strictly")
                else:
                    st.write("You are close! Focus on revision and mock tests")



    # ---------------- CHATBOT SCREEN ----------------

    elif st.session_state.app_page == "chatbot":

        st.title("🤖 AI Student Assistant")

        st.caption("Ask anything about the project, data science, ML, Python, and more!")

        SYSTEM_PROMPT = """


    You are the official AI assistant of the Student Performance System project.



    Answer questions about:


    1. This project and all its features


    2. Data Science concepts (what is data science, roadmap, career, tools)


    3. Python programming


    4. Machine Learning (algorithms, model building, evaluation)


    5. Libraries used: Streamlit, Scikit-learn, Pandas, Matplotlib, MySQL, NLTK



    PROJECT DETAILS:


    """ + project_info + """



    DATA SCIENCE KNOWLEDGE:


    - Python: interpreted, general-purpose language used for data science, ML, web dev


    - Data Science Roadmap: Python → Statistics → Pandas/NumPy → Visualization → ML → Deep Learning → Deployment


    - ML Model Building Steps: Collect data → Clean data → Feature Engineering → Train/Test Split → Scale → Train Model → Evaluate → Predict


    - Logistic Regression: classification algorithm that predicts probability of a class (Pass/Fail here)


    - StandardScaler: normalizes features so they have mean=0 and std=1


    - Accuracy: correct predictions / total predictions


    - Overfitting: model works well on training data but poorly on new data


    - Pandas: library for data manipulation using DataFrames


    - Matplotlib: library for creating charts and graphs


    - Scikit-learn: ML library with algorithms, preprocessing, evaluation tools



    Answer clearly, accurately, and helpfully. If asked about the project, explain in detail.


    If asked about data science or Python, give clear educational answers with examples.


    """

        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        user_query = st.chat_input("Ask anything...")

        if user_query:

            st.session_state.messages.append({"role": "user", "content": user_query})

            with st.chat_message("user"):

                st.markdown(user_query)

            try:

                with st.chat_message("assistant"):

                    with st.spinner("Thinking..."):
                        # Build conversation for HF

                        hf_messages = [{"role": "system", "content": SYSTEM_PROMPT}]

                        for m in st.session_state.messages:
                            hf_messages.append({"role": m["role"], "content": m["content"]})

                        completion = client.chat.completions.create(

                            model="Qwen/Qwen2.5-7B-Instruct",

                            messages=hf_messages,

                            max_tokens=1024

                        )

                        response_text = completion.choices[0].message.content

                        st.markdown(response_text)

                st.session_state.messages.append({"role": "assistant", "content": response_text})



            except Exception as e:

                st.error(f"⚠️ Chatbot Error: {e}")


    # ---------------- PREDICTION SCREEN ----------------

    elif st.session_state.app_page == "prediction":
        st.title("🎯 Student Performance Prediction")

        col1, col2, col3 = st.columns(3)

        with col1:
            study_hours = st.number_input("Weekly Study Hours", 0, 100)

        with col2:
            attendance = st.number_input("Attendance %", 0, 100)

        with col3:
            participation = st.number_input("Class Participation (1-5)", 1, 5)

        if st.button("Predict Result 🚀"):
            try:
                data = pd.read_csv("student_performance.csv")

                # ---------------------------------------------------
                # SMART TARGET
                # ---------------------------------------------------
                data['pass_fail'] = data.apply(
                    lambda row:
                    "Pass"
                    if (
                            (row['attendance_percentage'] >= 85 and row['class_participation'] >= 4)
                            or
                            (row['weekly_self_study_hours'] >= 5 and row['attendance_percentage'] >= 60)
                            or
                            (row['class_participation'] == 5 and row['attendance_percentage'] >= 50)
                    )
                    else "Fail",
                    axis=1
                )

                # ---------------------------------------------------
                # FEATURE ENGINEERING
                # ---------------------------------------------------
                data['study_attendance_combo'] = (
                        data['weekly_self_study_hours']
                        * data['attendance_percentage']
                )

                data['engagement_score'] = (
                        data['attendance_percentage']
                        + data['class_participation'] * 10
                )

                # ---------------------------------------------------
                # FEATURES
                # ---------------------------------------------------
                X = data[[
                    'weekly_self_study_hours',
                    'attendance_percentage',
                    'class_participation',
                    'study_attendance_combo',
                    'engagement_score'
                ]]

                y = data['pass_fail']

                # ---------------------------------------------------
                # TRAIN TEST SPLIT
                # ---------------------------------------------------
                x_train, x_test, y_train, y_test = train_test_split(
                    X,
                    y,
                    test_size=0.2,
                    random_state=42
                )

                # ---------------------------------------------------
                # SCALING
                # ---------------------------------------------------
                scaler = StandardScaler()

                x_train = scaler.fit_transform(x_train)
                x_test = scaler.transform(x_test)

                # ---------------------------------------------------
                # LOGISTIC REGRESSION MODEL
                # ---------------------------------------------------
                model = LogisticRegression()

                model.fit(x_train, y_train)

                # ---------------------------------------------------
                # SAVE MODEL + SCALER
                # ---------------------------------------------------
                joblib.dump(model, "model.pkl")
                joblib.dump(scaler, "scaler.pkl")

                # ---------------------------------------------------
                # LOAD MODEL
                # ---------------------------------------------------
                model = joblib.load("model.pkl")
                scaler = joblib.load("scaler.pkl")

                # ---------------------------------------------------
                # PREDICTION INPUT
                # ---------------------------------------------------
                new_student = pd.DataFrame(
                    [[study_hours, attendance, participation]],
                    columns=[
                        'weekly_self_study_hours',
                        'attendance_percentage',
                        'class_participation'
                    ]
                )

                # ---------------------------------------------------
                # SAME FEATURE ENGINEERING
                # ---------------------------------------------------
                new_student['study_attendance_combo'] = (
                        new_student['weekly_self_study_hours']
                        * new_student['attendance_percentage']
                )

                new_student['engagement_score'] = (
                        new_student['attendance_percentage']
                        + new_student['class_participation'] * 10
                )

                # ---------------------------------------------------
                # SAME COLUMN ORDER
                # ---------------------------------------------------
                new_student = new_student[[
                    'weekly_self_study_hours',
                    'attendance_percentage',
                    'class_participation',
                    'study_attendance_combo',
                    'engagement_score'
                ]]

                # ---------------------------------------------------
                # APPLY SCALING
                # ---------------------------------------------------
                new_student_scaled = scaler.transform(new_student)

                # ---------------------------------------------------
                # PREDICTION
                # ---------------------------------------------------
                pred = model.predict(new_student_scaled)

                probability = model.predict_proba(new_student_scaled).max() * 100

                # ---------------------------------------------------
                # RESULT OUTPUT
                # ---------------------------------------------------
                if pred[0] == "Pass":
                    st.success(f"🎉 Result: PASS ")
                else:
                    st.error(f"❌ Result: FAIL ")

            except FileNotFoundError:
                st.error("❌ Dataset file not found")

            except ValueError:
                st.error("❌ Invalid input values")

            except KeyError as e:
                st.error(f"❌ Missing column in dataset: {e}")

            except Exception as e:
                st.error(f"⚠️ Prediction Error: {e}")
