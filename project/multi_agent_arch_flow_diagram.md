## Multi-Agent Workflow
 
I designed the agent flow as below based on how a customer request moves through the system.
 
Customer Request  
↓  
**Orchestrator Agent**  
- Receives the request
- Identifies what needs to be done
- Coordinates the other agents
 
↓  
 
**Inventory Agent**  
- Checks the requested paper and quantity
- Checks current inventory
- If stock is low, checks supplier delivery timeline
 
↓  
 
**Quote Agent**  
- Checks previous quote information
- Calculates the quote
- Applies pricing/discount logic where required
 
↓  
 
**Sales Agent**  
- Checks whether the order can be completed
- Checks required financial information
- Fulfills the order when possible
 
↓  
 
**Final Decision**
 
If order can be fulfilled → Complete order → Update inventory/financial data  
If order cannot be fulfilled → Keep order unfulfilled → Return the reason  
 
↓  
 
**Final response to customer**

### Workflow Note
 
I initially planned the agents around the main business activities: inventory, quotation and sales. While implementing the solution, I kept the orchestrator as the main entry point so the customer request does not directly interact with each agent.
 
The inventory check happens before the final sales decision because the system needs to know whether the requested quantity is available. If there is not enough stock, supplier delivery information is checked before deciding how the request should be handled.
 
This flow also made it easier for me to keep each agent focused on one responsibility instead of putting all the business logic into a single agent.