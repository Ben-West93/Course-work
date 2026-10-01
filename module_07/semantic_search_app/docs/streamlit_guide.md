# Streamlit Guide

Streamlit is a Python library for turning scripts into interactive web applications without writing any HTML, CSS, or JavaScript. You write ordinary Python, call functions such as st.title, st.write, and st.button, and Streamlit renders them in the browser. Start an app from the terminal with streamlit run app.py.

The most important concept in Streamlit is the re-run model. Every time a user interacts with any widget, the entire Python script re-runs from top to bottom. Widgets return their current value on each run, so a text input simply returns whatever the user has typed so far. This model is simple to reason about but has consequences for state and performance.

Because scripts re-run completely, regular variables reset on every interaction. Use st.session_state to persist data across re-runs. The initialization pattern is: if "key" not in st.session_state: st.session_state["key"] = default_value. Widgets created with a key argument store their value in session state automatically, and callbacks passed through on_click or on_change run before the next re-run, which is the safe place to modify widget values.

Expensive work such as loading a machine learning model or opening a database connection should not happen on every re-run. The st.cache_resource decorator stores the return value of a function once per server process and shares it across all sessions and re-runs. For data such as DataFrames or API responses, st.cache_data caches a serializable copy keyed on the function arguments.

Streamlit provides layout components like st.columns for side-by-side content, st.sidebar for controls, st.tabs for organizing views, st.container for grouping elements, and st.expander for collapsible sections. Use st.set_page_config at the top of your script for page-wide settings such as the browser tab title, icon, and a wide layout.

When you need to restart the script immediately, for example after updating session state from a button, call st.rerun(). Feedback elements such as st.success, st.info, st.warning, and st.error display colored message boxes, and st.metric shows a large number with an optional delta, which is useful for dashboards and sidebar statistics.
