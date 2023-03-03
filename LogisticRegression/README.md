You can execute the Python as any Python script. First, try running the following:

```
python3 LogisticWorkFlow.py
```

This will validate the flow structure without executing any steps. Metaflow has a number of rules for what is considered a valid DAG.

![](https://github.com/mmuratarat/metaflow_example/blob/main/LogisticRegression/screenshots/Screenshot%202023-03-03%20at%209.33.49%20PM.png)

Metaflow also runs a basic code check, a linter, every time you execute the script, which can detect typos, missing functions, and other such syntactic errors. If any issues are found, an error is shown and nothing else is run.

Now try running the next code:

```
python3 LogisticWorkFlow.py show
```

This should print out a textual representation of the DAG.

![](https://github.com/mmuratarat/metaflow_example/blob/main/LogisticRegression/screenshots/Screenshot%202023-03-03%20at%209.31.42%20PM.png)

Now, let’s execute the flow, as shown next! We call an execution of a flow *a run*:

```
python3 LogisticWorkFlow.py run
```

This command executes all the steps defined in the DAG in order. If all goes well, you should see a bunch of lines printed out that look like this:

![](https://github.com/mmuratarat/metaflow_example/blob/main/LogisticRegression/screenshots/Screenshot%202023-03-03%20at%209.32.02%20PM.png?raw=true)
