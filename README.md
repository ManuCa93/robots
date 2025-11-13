# Project: Garbage Sorting with a Robot Arm

## **Goal**

Develop and compare two distinct control strategies for a **Panda robot manipulator** to perform an automated garbage sorting task. The robot must identify objects (e.g., colored cubes) and sort them into designated bins based on their color.

## **Project Objective**

The core objective is to create an algorithm that successfully calculates the picking and placing configurations for the robot arm for multiple objects and executes the sorting task using two different motion control strategies.

---

## **Implementation Details**

### **The Robot System**

* **Manipulator:** Utilize the **Panda robot manipulator**.
* **Assumption:** The color of the objects is assumed to be known (focus is on motion control).
* **Object Identification & IK:** For each detected object $i$:
    * Compute the **picking position** $P_{\text{pick},i}$ and the **placing position** $P_{\text{place},i}$ based on the object’s color.
    * Use **Inverse Kinematics (IK)** to obtain the robot joint configurations $q_{\text{pick},i}$ and $q_{\text{place},i}$.

### **Control Strategies to Compare**

The sorting task will be executed using the following two robot control methods:

1.  **Joint-Space Trajectory Interpolation**
    * **Method:** Move from $q_{\text{pick},i}$ to $q_{\text{place},i}$ using interpolated joint trajectories.
    * **Focus:** Smooth and efficient movement in the joint space.

2.  **Resolved-Rate Motion Control (RRMC)**
    * **Method:** Use Cartesian velocity control (e.g., Jacobian pseudo-inverse) to smoothly reach the pick and place poses in the operational (Cartesian) space.
    * **Focus:** Direct control over the end-effector's path.

---

## **Analysis and Visualization**

A crucial part of the project is the comparative analysis of the two control strategies.

### **Analysis Tasks**

* Plot and compare trajectories for both strategies in **joint space** and **Cartesian space**.
* Analyze the **end-effector path smoothness**.
* Evaluate the **execution time** for the complete sorting task.
* Assess the overall **sorting accuracy** (e.g., proximity to the desired placement location).

### **Visualization Requirements**

* **Plots/Graphs:** Clear, informative, and well-labeled visualizations (e.g., time vs. joint angles, time vs. end-effector position/velocity).
* **Execution Demo:** Include a video or live demo of the robot successfully executing the sorting task.

---

## **Project Structure and Timeline**

| WEEK / DATE | ACTIVITY | Notes |
| :---: | :--- | :--- |
| **\#9 / Today** | **Project Proposal Presentation & Group formation** | Initial concept submission. |
| **\#10 / 21 November** | Project Development | Coding and implementation work. |
| **\#11 / 28 November** | **Project Intermediate Presentation** | Mandatory check-in (not evaluated). |
| **\#12 / 5 December** | Project Development | Refinement, analysis, and visualization. |
| **\#13 / 12 December** | **Presentation Project & Evaluation (!!!)** | Final oral presentation and submission. |

---

## **Evaluation Criteria Summary**

| Category | Weight |
| :--- | :---: |
| **Technical Implementation** | 35% |
| **Analysis & Interpretation** | 20% |
| **Visualization & Communication**| 20% |
| **Presentation Skills** | 15% |
| **Teamwork & Organization** | 10% |

---

## **Final Presentation Structure**

1.  **Introduction of the Task:** Define the sorting objective and problem statement.
2.  **Method Explanation:** Detail the IK and the two control strategies (Joint-Space and RRMC).
3.  **Execution Demo (or video):** Showcase the successful operation of the code/robot.
4.  **Plot Analysis:** Present key take-away messages and comparison results.
5.  **Conclusion:** Summarize findings and future work.

---

## **Team**

* **Group Members:**
    * Manuel Cattoni
    * Alessio Carnevale
    * Carlo Schillaci
