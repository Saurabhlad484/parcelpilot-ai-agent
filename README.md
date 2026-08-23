# ParcelPilot AI Agent

An authority-aware RAG-based customer support assistant with structured data tools, document retrieval, access control, and proactive issue detection.

## Live Demo

 **[Click here to try the ParcelPilot AI Agent](https://parcelpilot-ai-agent-8ghaeogsg8ofnbgpgh2bhl.streamlit.app)**

##  Project Overview

ParcelPilot AI Agent is an AI-powered support assistant designed to answer customer and internal support queries using structured data and authoritative documents.

The system is designed to retrieve the appropriate evidence before generating an answer, while considering:

- Customer-specific agreements
- Current governing policies and SOPs
- Deprecated policy handling
- Account-level access control
- Structured order, account, and ticket data
- Historical support information as non-authoritative context
- Proactive issue detection

##  Key Features

### Access Control
Implements a simple role-based access control (RBAC) model.

- Support agents can access all accounts.
- Customers can access only their authorized account.
- Account-level access is checked before accessing protected information.

##  Sample Questions

Try asking the ParcelPilot AI Agent questions like:

###  Order & Account Queries
- What is the status of order ORD-1001?
- Who is the account owner for order ORD-1001?
- Show me all orders for account ACCT-001.
- Show me the support tickets for account ACCT-002.
- What information do we have for ticket TKT-502?

###  Cancellation Queries
- Can ORD-1001 be cancelled without a fee?
- What are the cancellation terms for ACCT-001?
- How long after booking was the cancellation requested for ORD-1001?

### Failed Pickup & Service Credit
- Does ORD-2002 qualify for a service credit after a failed pickup?
- How long has ORD-2002 remained without pickup?
- What are the failed pickup service credit terms for ACCT-002?

### Support SLA Queries
- What is the P1 response target?
- What are the current support response targets?
- Does ACCT-001 have a custom support SLA?

###  Product & Known Issues
- What is the bulk upload limit?
- Why does TKT-502 fail when uploading a 4200-row CSV?
- Is there a known issue related to SwiftShip webhook confirmation delays?

### Proactive Issue Detection
- Show me the current open support issues.
- Are there any recurring customer issues?
- Are there similar issues affecting multiple customers?
- Show me any unusual support patterns.

### Structured Data Tools
The application can retrieve and analyze:

- Account information
- Order information
- Support tickets
- Orders belonging to an account
- Tickets belonging to an account
- Cancellation timing
- Failed pickup timing
- Dataset snapshot information

### Authority-Aware Document Retrieval

The system follows an evidence hierarchy:

1. Customer agreement, when relevant to the topic
2. Current governing policy, SOP, or operational guide
3. Historical support information as non-authoritative context
4. Deprecated documents only when explicitly requested

###  RAG-Based Document Search

The document pipeline includes:

```text
PDF Documents
      ↓
Document Processing
      ↓
Page-Level Text Extraction
      ↓
Semantic Chunking
      ↓
OpenAI Embeddings
      ↓
ChromaDB Vector Store
      ↓
Authority-Aware Retrieval
      ↓
Grounded AI Response
