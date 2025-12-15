28/nov/2025 / Day 1:
Change: 
- Adding .gitignore, data_loader.py, main.py, preprocess.py, requirement.txt, window.py.
Expectation:
- preprocess data to be ready to use for the model.
- know the shape of data for multiplication with weight.
Notes: 
- there's a room for improvement in normalizing data.

1/dec/2025 / Day 4:
Change:
- Remove most of requirement.txt.
Expectation:
- To make this project easiear to understand.

6/dec/2025 / Day 9:
Change:
- Adding normalization function.
Expectation:
- Easier to read code
Result:
- All of data right now is in interval 0-1 this make the model taking a small step in gradient descent so that they know the lowest cost.
- Improve model accuracy.

7/dec/2025 / day 10:
Change:
- Change variable from data_scaled - data_normal
Expectation:
- Easier to read code

11/dec/2025 / Day 14:
Change:
- Add __init__ in model.py
Expectation:
- preparation for feed forward model

14/dec/2025 / Day 16:
Change:
- Fixing data leakage inside preprocess.py and window.py
Notes:
- This is just log you can see the detail in my post-mortem.

16/dec/2025 / Day 18:
Change:
- Adding experiment_log.
Expectation:
- To see how far the progress.