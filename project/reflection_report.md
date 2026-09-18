# Multi-Agent Paper Supply System – Reflection Report
 
## 1. System Overview
 
The goal of this project was to build a multi-agent system that can handle different types of customer requests for paper supplies. Instead of having one component do everything, I divided the work between different agents. Each agent is responsible for a specific part of the order process, such as checking inventory, preparing a quote, checking supplier delivery dates, and completing the sale.
 
An orchestrator coordinates these agents and makes sure the required information moves between them in the correct order.
 
The system mainly handles the following activities:
 
- Reads and understands the customer's request
- Checks whether the requested item and quantity are available
- Identifies when additional inventory is required
- Checks supplier delivery dates when stock is insufficient
- Prepares the quote for the requested items
- Completes orders when the required conditions are met
- Leaves an order unfulfilled when it cannot be completed and provides a reason
- Keeps track of inventory and financial information as requests are processed
 
## 2. Multi-Agent Workflow
 
The customer request first goes to the orchestrator, which manages the overall flow and coordinates the other agents.
 
The general flow I followed is:
 
**Customer Request → Orchestrator → Inventory Agent → Quote Agent → Sales Agent → Final Response**
 
The Inventory Agent checks whether enough stock is available. If more inventory is needed, the system also checks the expected supplier delivery date before deciding whether the request can be completed.
 
The Quote Agent prepares the quote using the information gathered during the inventory check, and the Sales Agent handles the final order decision.
 
Keeping these responsibilities separate made the overall logic easier to follow and also helped while testing and debugging individual parts of the system.
 
## 3. Evaluation Results
 
I tested the completed system using the provided `quote_requests_sample.csv` file. A total of **20 customer requests** were processed, and the results were saved into `test_results.csv`.
 
During testing, I was able to see both successful and unsuccessful scenarios. Multiple customer orders were completed successfully, which satisfies the requirement of completing at least three orders.
 
The system also correctly left some requests unfulfilled instead of treating every request as successful. For example, when the requested paper type could not be identified, the system returned an `item_not_found` status along with an explanation.
 
The testing also helped confirm that inventory and financial information continued to be maintained while multiple requests were processed.
 
## 4. What Worked Well
 
One thing that worked well was separating the responsibilities between the agents. The Inventory Agent focuses on stock availability, while the Quote and Sales Agents handle their own parts of the process. This made the workflow easier for me to understand and troubleshoot.
 
The system also checks inventory before completing an order. When there is not enough inventory, it checks the supplier delivery information instead of immediately completing the transaction.
 
Another useful behavior is that the system does not automatically accept every request. If a request cannot be understood or fulfilled, it returns an appropriate status and reason. This was important during testing because the project requires both successful and unsuccessful order scenarios.
 
Saving all 20 results to `test_results.csv` also gave me a simple way to review how the system behaved across the full test dataset.
 
## 5. Areas for Improvement
 
One area I noticed during testing is customer-request interpretation. Some requests resulted in `item_not_found` because the system could not identify the requested paper type. The parsing logic could be improved so that different ways of describing the same product can be recognized more consistently.
 
The generated test results could also be easier to review. Currently, the response contains detailed nested information from the agents. It works, but having separate columns for values such as final status, fulfilled/unfulfilled, failure reason, and expected delivery date would make the results much easier to analyze.
 
## 6. Future Improvements
 
### Better Request Understanding
 
I would improve the request-parsing logic so that common variations in product names can be mapped to the correct inventory item. This should reduce cases where a valid customer request is marked as `item_not_found`.
 
### Better Evaluation Reporting
 
I would also add a small summary to the evaluation showing values such as the total number of requests, completed orders, unfulfilled orders, success rate, inventory shortages, and requests that could not be understood. This would make it easier to evaluate the system without manually reviewing every row in the CSV.
 
### Smarter Inventory Planning
 
Another improvement would be to use previous order patterns to predict when inventory is likely to run low. Instead of waiting for a customer request to expose a shortage, the system could identify frequently requested products and recommend reordering them earlier.
 
## 7. Conclusion
 
This project helped me understand how a larger business process can be divided among multiple agents instead of putting all the logic into one place. The final system coordinates inventory checking, quote preparation, supplier delivery checks, and sales processing while keeping track of the inventory and financial state.
 
Testing the system with the provided dataset was also useful because it exposed different scenarios rather than only successful orders. The system was able to complete valid requests and leave requests unfulfilled when necessary with a reason.
 
There are still improvements I would make, especially around request parsing and reporting, but the current implementation provides a working multi-agent flow that meets the main goals of the project.
 