# Association Rule Learning: Business Problem and Dataset

Association rule learning discovers items or events that frequently occur together. Its best-known form is:

> Customers who bought \(X\) also bought \(Y\).

This type of analysis is especially useful in retail and e-commerce, where businesses record large collections of shopping baskets and want to understand relationships among products.

## How Association Rule Learning Differs

Association rule learning is different from the methods introduced earlier:

| Method | Main objective |
|---|---|
| Regression | Predict a continuous target |
| Classification | Predict a known category |
| Clustering | Discover groups of similar observations |
| Association rule learning | Discover recurring co-occurrence relationships among items |

Association rule learning has no predefined dependent variable. It searches transaction data for rules that relate one itemset to another.

Clustering also does not create a true dependent variable in the supervised-learning sense. It produces derived cluster labels. Association rule learning instead produces rules such as:

\[
\{\text{Burger}\}
\rightarrow
\{\text{French Fries}\}
\]

The rule means that fries occur frequently enough among baskets containing burgers to satisfy the selected rule-quality thresholds. It does not prove that buying a burger causes a customer to buy fries.

## Apriori and Eclat

This part of the course introduces two association-mining techniques:

- **Apriori**, which generates frequent itemsets level by level and prunes candidates using the Apriori principle; and
- **Eclat**, which typically uses transaction-ID intersections to calculate itemset support.

The practical implementation begins with Apriori in the `apriori.ipynb` notebook.

## Business Scenario

Imagine a food retailer that wants to improve product offers, bundles, recommendations, and merchandising. The retailer collects its completed baskets and asks a data scientist to identify strong product associations.

The business question is:

> Which products occur together often enough—and with enough association strength—to support a useful commercial action?

Potential actions include:

- recommending a complementary product;
- creating a discounted bundle;
- designing a cross-sell message;
- organizing store or website navigation;
- planning stock for associated products; and
- testing a targeted promotion.

The analysis should not automatically translate every rule into a “buy one, get one free” offer. If customers already purchase the consequent without an incentive, giving it away may reduce margin without increasing demand. The correct offer must consider incremental sales, product costs, margins, inventory, and customer value.

## The Market Basket Dataset

The file `Market_Basket_Optimisation.csv` contains **7,501 transaction rows**.

Each row represents one basket or checkout transaction. Each populated cell contains an item purchased in that transaction.

Conceptually:

| Transaction | Purchased items |
|---|---|
| \(T_1\) | Shrimp, almonds, avocado, vegetable mix, green grapes, ... |
| \(T_2\) | Burgers, meatballs, eggs |
| \(T_3\) | Chutney |
| \(T_4\) | Turkey, avocado, ... |

Basket sizes vary. The CSV represents them in a rectangular table, so shorter rows contain empty trailing cells when imported.

> A row identifies a transaction, not necessarily a unique customer. Without a customer identifier, the data cannot establish that every row came from a different person or connect several visits made by the same customer.

## Transaction Data versus a Conventional Feature Table

In supervised learning, rows usually represent observations and columns have fixed meanings such as age, income, or price.

The market-basket file is different:

- rows are transactions;
- populated values are item names;
- the position of an item within a row is not usually meaningful;
- different transactions contain different numbers of items; and
- there is no target column.

The analysis therefore converts each row into an item collection:

\[
T_i=\{\text{items purchased in transaction }i\}
\]

Item order should not be interpreted as a purchase sequence unless timestamps or sequence information are explicitly available.

## The Desired Output

An association rule has the form:

\[
X\rightarrow Y
\]

where:

- \(X\) is the **antecedent**;
- \(Y\) is the **consequent**; and
- \(X\cap Y=\varnothing\).

For a two-product rule:

\[
\{\text{Item A}\}
\rightarrow
\{\text{Item B}\}
\]

the analysis asks whether transactions containing Item A contain Item B more often than expected.

Rules can also contain multiple items:

\[
\{\text{Burger, French Fries}\}
\rightarrow
\{\text{Ketchup}\}
\]

However, larger itemsets increase the search space and may be harder to explain or operationalize. This exercise focuses mainly on interpretable two-item relationships.

## Evaluating Candidate Rules

Strong rules are not selected using co-occurrence counts alone. The main measures are:

### Support

Rule support measures how often the full itemset appears:

\[
\operatorname{support}(X\rightarrow Y)
=
\operatorname{support}(X\cup Y)
=
\frac{\#(X\cup Y)}{N}
\]

### Confidence

Confidence measures how often \(Y\) appears among transactions containing \(X\):

\[
\operatorname{confidence}(X\rightarrow Y)
=
\frac{\operatorname{support}(X\cup Y)}
{\operatorname{support}(X)}
\]

### Lift

Lift compares the rule confidence with the baseline frequency of \(Y\):

\[
\operatorname{lift}(X\rightarrow Y)
=
\frac{\operatorname{confidence}(X\rightarrow Y)}
{\operatorname{support}(Y)}
\]

Interpretation:

- lift \(>1\): positive association;
- lift \(=1\): no observed association; and
- lift \(<1\): negative association.

A useful rule normally needs adequate support, meaningful confidence, lift above the chosen requirement, and practical business value.

## From Association to Recommendation

Rules can support recommendation features such as “frequently bought together.” However, an association rule does not predict with certainty what an individual customer will purchase.

Large e-commerce platforms generally use more sophisticated recommender systems that may combine:

- collaborative filtering;
- customer and product features;
- browsing and purchase sequences;
- contextual signals;
- ranking models; and
- controlled experimentation.

Apriori provides an interpretable baseline for item co-occurrence, not a complete description of a modern production recommender.

## A Better Analytical Mission

The data scientist's mission is to:

1. load each row as a transaction;
2. discover sufficiently frequent itemsets;
3. generate directional association rules;
4. filter and rank rules using support, confidence, and lift;
5. remove trivial, redundant, unstable, or unactionable results;
6. translate promising rules into business hypotheses; and
7. test those hypotheses before deployment.

The goal is not merely to return the rule with the largest metric value. It is to identify relationships that are reliable, interpretable, profitable, ethical, and actionable.

---

# Study Notes

## Core idea

> Association rule learning analyzes baskets to discover products that occur together more often than expected.

## Dataset facts

| Property | Value |
|---|---|
| File | `Market_Basket_Optimisation.csv` |
| Number of rows | 7,501 |
| Unit of observation | One transaction/basket |
| Values | Product names |
| Basket length | Variable |
| Target variable | None |
| Customer identifier | Not present |

## Important terminology

| Term | Meaning |
|---|---|
| Transaction | One observed basket |
| Item | A product appearing in a transaction |
| Itemset | A collection of one or more items |
| Antecedent | Left side of a rule |
| Consequent | Right side of a rule |
| Association rule | Directional implication \(X\rightarrow Y\) |
| Support | Frequency of an itemset in all transactions |
| Confidence | Conditional frequency of \(Y\) among baskets containing \(X\) |
| Lift | Confidence relative to the baseline frequency of \(Y\) |

## Association is not prediction or causation

A rule:

\[
X\rightarrow Y
\]

does not mean:

- every customer buying \(X\) will buy \(Y\);
- \(X\) causes the purchase of \(Y\);
- \(X\) was purchased before \(Y\); or
- offering \(Y\) for free will increase profit.

It describes a relationship observed in the transaction sample.

## Direction matters

\[
\operatorname{confidence}(X\rightarrow Y)
\neq
\operatorname{confidence}(Y\rightarrow X)
\]

The reverse rule answers a different question because its denominator changes. Both directions should be evaluated separately.

## Practical workflow

\[
\text{Raw baskets}
\rightarrow
\text{transaction lists}
\rightarrow
\text{frequent itemsets}
\rightarrow
\text{association rules}
\rightarrow
\text{metric filtering}
\rightarrow
\text{business review}
\rightarrow
\text{experiment}
\]

## Data-quality checks

Before mining rules:

- normalize inconsistent product names;
- remove accidental whitespace and duplicate entries within a basket;
- distinguish missing cells from actual product values;
- verify whether returns or cancelled orders are present;
- check the collection period and store coverage;
- investigate unusual bulk or staff transactions;
- decide how promotions should be represented; and
- confirm whether repeated transactions from the same customer can be linked.

## Business checks

Before turning a rule into an offer:

- calculate the number of supporting baskets;
- compare product margins and promotion costs;
- check whether the products are substitutes or complements;
- determine whether an existing promotion created the rule;
- examine seasonality and location effects;
- assess stock and fulfillment constraints;
- estimate whether the action creates incremental purchases;
- consider customer convenience and fairness; and
- validate performance with a control group.

## Why “buy one, get one free” can be risky

If \(X\rightarrow Y\) already has high confidence, many customers purchasing \(X\) may have bought \(Y\) at full price anyway. Giving \(Y\) away can cannibalize existing revenue.

Alternatives may include:

- a partial bundle discount;
- a recommendation without a discount;
- a threshold-based offer;
- loyalty points;
- a trial-sized product; or
- a test limited to customers unlikely to purchase \(Y\) without intervention.

## Train/test thinking still matters

Association rule mining is unsupervised, so it does not require the conventional feature/target split used in supervised learning. Nevertheless, rules should be validated on a later time period or independent transaction sample.

A rule found in one week may reflect temporary promotions, weather, holidays, or random variation. Out-of-time validation helps determine whether it persists.

## Common misconceptions

### “Every row represents a different customer”

Not necessarily. Each row is a transaction, and the file contains no customer identifier.

### “Association rules are ordinary correlations”

They are co-occurrence rules evaluated with measures such as support, confidence, and lift. They do not automatically provide a conventional correlation coefficient.

### “Amazon recommendations are simply Apriori”

Large platforms generally use multiple, more sophisticated recommendation and ranking methods. Apriori is an interpretable foundational technique.

### “The rule with the highest confidence is the best deal”

High confidence may result from a very popular consequent. Lift, support, stability, margin, and incremental impact also matter.

### “A free consequent guarantees more profit”

Profit depends on incremental demand, margins, costs, substitution effects, and customer response.

## Questions for self-testing

1. What is the main objective of association rule learning?
2. How does it differ from classification and clustering?
3. What does one row of the market-basket dataset represent?
4. Why should a row not automatically be called a unique customer?
5. Why are trailing cells empty in some rows?
6. What are the antecedent and consequent of a rule?
7. How do support, confidence, and lift answer different questions?
8. Why can the confidence of a rule differ from that of its reverse?
9. Why does an association not prove causation or purchase order?
10. Why might giving away the consequent reduce profit?
11. What should be checked before a rule becomes a promotion?
12. Why should rules be validated on later transactions?
13. How can a rule be useful without offering a discount?
14. Why is Apriori only one possible recommender-system component?
