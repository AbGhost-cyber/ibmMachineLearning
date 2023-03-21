import pandas as pd
import numpy as np

df = pd.DataFrame({
    'name': ['Amy', 'Jackie', 'Sue'],
    'grades': [90, 84, 76]
})

# use the lambda function to multiply bump up the values in the grades column by 1.2!
df['grades'] = df['grades'].apply(lambda x: x - 1.2)
mList = [90, 84, 76]
data = sorted(mList)
print(data)


class ClassSchedule:

    def __init__(self):
        self._courses = []

    def add_new_course(self, course):
        self._courses.append(course)
        return self

    def print_courses(self):
        print(f"you have {len(self._courses)} courses in total")
        for course in self._courses:
            print(course)


schedule = ClassSchedule()
mInput = input("Input a course\n")
while mInput != "show":
    schedule.add_new_course(mInput)
schedule.print_courses()

if __name__ == '__main__':
    schedule.print_courses()
