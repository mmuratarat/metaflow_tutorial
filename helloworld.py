from metaflow import FlowSpec, step

class HelloWorldFlow(FlowSpec):

    @step
    def start(self):
        """
        Starting point
        """
        print("this is the start step")
        self.next(self.hello)

    @step
    def hello(self):
        """
        Just saying hello
        """
        print("Hello world!")
        self.next(self.end)

    @step
    def end(self):
        """
        Finish line
        """
        print("This is the end step")

if __name__ == "__main__":
    HelloWorldFlow()
