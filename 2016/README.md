<p align="center">
  <img src="../assets/katana-graph-logo.png" alt="KatanaGraph logo" width="320">
</p>

# KatanaGraph Developers Notebook Archive 2016

<table>
  <tr>
    <td><a href="../README.md"><strong>Archive Home</strong></a></td>
    <td><strong>2016</strong></td><td><a href="../2017/README.md"><strong>2017</strong></a></td>
  </tr>
</table>


This year view preserves topic folders from the KatanaGraph notebook archive and maps them onto a monthly article cadence, matching the archive style used for the MongoDB developers notebook site.

### [February 2016: Connection and Graph Lifecycle](../10_Connection/README.md)

**Customer:** I am getting started with KatanaGraph and need the basic operational flow. How do I connect to the service, validate pod placement, and create or drop graphs cleanly from notebooks?

**Daniel:** This topic folder walks through the connection bootstrap, graph create and drop patterns, and a lightweight environment check for pod placement so you can validate the platform before moving on to analytics work.

### [March 2016: Importing Data into KatanaGraph](../20_ImportData/README.md)

**Customer:** I have files and data frames ready to go, but the import path is not obvious. What is the right way to load bucket data, run mini-batch ingestion, and manage multiple node labels in KatanaGraph?

**Daniel:** These notebooks focus on ingestion mechanics, including bucket-based loading, incremental and mini-batch patterns, and variations in node-label modeling so you can move raw source data into a working graph quickly.

### [April 2016: First Steps with Cypher](../22_FirstStepsCypher/README.md)

**Customer:** My team knows SQL better than graph query languages. What does the KatanaGraph Cypher workflow look like for first traversals, dictionary and array handling, cleanup operations, and directional edge patterns?

**Daniel:** This folder introduces the everyday Cypher basics: setup, traversals, collection handling, graph cleanup, and practical update patterns that help a new user become productive with the query language.

### [May 2016: REST API Basics](../23_Rest%28V0%29/README.md)

**Customer:** We may need to drive KatanaGraph from external services instead of only through notebooks. What does the early REST interface look like, and what is the minimum setup for exercising it?

**Daniel:** The notebooks here show a first-pass REST workflow, pairing environment setup with direct service calls so you can test remote control patterns before investing in a larger application wrapper.

### [June 2016: Path Analytics](../A1_Path/README.md)

**Customer:** I need route and traversal analytics over connected data. How do I set up sample graphs and run shortest path, breadth-first search, KSSSP, and random-walk style analysis in KatanaGraph?

**Daniel:** This topic collects the path-oriented analytics material, using airport and related sample graphs to explore traversal setup, SSSP, BFS, KSSSP, and other path-centric algorithms.

### [July 2016: Community Detection](../A2_Community/README.md)

**Customer:** My graph has clusters that I need to expose for analysis and segmentation. How do I prepare representative data sets and run Louvain-style community detection in KatanaGraph?

**Daniel:** The notebooks here move from setup into community analytics, including multiple graph sizes and environments, so you can see how KatanaGraph surfaces clusters and group structure in connected data.

### [August 2016: Centrality Analytics](../A3_Centrality/README.md)

**Customer:** I want to identify the most influential nodes in several graphs. What is the KatanaGraph workflow for setting up sample data and computing betweenness and PageRank style centrality measures?

**Daniel:** This folder focuses on centrality routines and shows how to prepare several data sets, then run ranking and influence-oriented analytics such as betweenness and PageRank.

### [September 2016: Houston Use Case](../AH_Houston/README.md)

**Customer:** Can KatanaGraph support a location-specific analytical use case with realistic data rather than only toy examples? I want to see a concrete Houston-centered scenario with graph queries and results.

**Daniel:** These notebooks build a Houston-oriented example that combines setup with applied Cypher work, giving you a more concrete end-to-end use case than the smaller tutorial graphs.

### [October 2016: LDBC and Routines](../AI_LdbcAndRoutines_Justin/README.md)

**Customer:** We want to benchmark and experiment with richer graph structures. How do I load LDBC-style data and combine it with routines like PageRank, Louvain, and betweenness in KatanaGraph?

**Daniel:** This topic folder covers loading the LDBC benchmark-style data set and pairing it with algorithmic routines, making it useful for larger-scale experiments and platform evaluation work.

### [October 2016: Core Notebook Compulsories](../C1_Compulsaries/README.md)

**Customer:** Before I get fancy, I need the repeatable basics. What are the core notebook patterns for display options, dataframe reshaping, graph import, versioning, saving RDG artifacts, and fetching results?

**Daniel:** This is the utility belt for the repository: display setup, dataframe transformation, import helpers, versioning notes, save and fetch patterns, and other foundational notebook techniques used throughout the project.

### [November 2016: User-Defined Functions](../C2_UDF/README.md)

**Customer:** I need logic that goes beyond stock queries and built-in analytics. How do I set up and test KatanaGraph UDF workflows, from simple hello-world patterns to graph mutation and MPI-oriented experiments?

**Daniel:** The UDF notebooks cover setup, implementation patterns, iterative experiments, and load tests so you can extend KatanaGraph behavior with custom logic rather than staying limited to built-in operations.

### [December 2016: REST and GraphQL Services](../C3_RestGraphQL/README.md)

**Customer:** We are considering an application layer in front of KatanaGraph. What does a service-oriented pattern look like when combining setup, a simple client, a web server, and GraphQL-style schema wiring?

**Daniel:** This folder packages KatanaGraph access into early service examples, showing how REST and GraphQL ideas fit together with lightweight Python components and notebook-driven setup.
