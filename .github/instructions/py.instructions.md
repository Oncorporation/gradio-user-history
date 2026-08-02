## Python and Streamlit Instructions
---
applyTo: `**/*.py`
---

- Write clear and concise docstrings for each function and class.
- Use snake_case for function names, variable names, and module names.
- Use CamelCase for class names.
- Follow PEP 8: Use 4 spaces for indentation, limit lines to 79 characters.
- Add a blank line before and after function definitions.
- For Streamlit: Prefix components with `st.`, organize UI elements logically, use `st.sidebar` for controls.
- Ensure imports are at the top, grouped by standard, third-party (e.g., streamlit), then local.
- never loop the same command to burn user tokens, ask if you run into an error for permission
- in html string variables use double curly braces for interpolation, e.g., `{{ variable_name }}`, especially in f-strings with "<script>" tags.
