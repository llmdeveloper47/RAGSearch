### NOTE : 

To Run Test Cases, please cd into the root folder and then first run the following command

 - STEP1 - export PYTHONPATH=$(pwd)/src

Then run the following from the root folder

 - STEP2 - pytest tests/

### Load Testing : 

To load test the application, there are two tests applied :
 - test_cli_with_json_input.py which loads the query-answer pairs from evaluation folder and tests the application on that
 - locust.py which load tests the application in browser with different number of users and more confguration options

To run the locust file, please cd into the root folder , then run locust -f tests/locustfile.py
this will open a locust session on the browser or give a local host path to test the application.

