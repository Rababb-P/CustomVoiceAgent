---
source: projects/reparo
topic: projects
last_updated: 2026-07-05
---

# Reparo

## What it is

Reparo is an agentic AI pipeline I built in 36 hours at Hack Canada 2025. You
describe a broken household item in natural language, it diagnoses the problem,
identifies the exact repair part you need, and one-click checks it out from
live Shopify inventory.

## The outcome

It won the $5,000 Reactiv prize and was a finalist for Most Complex AI Hack.

## How it works

It's a multi-step agent: diagnosis, then part identification, then catalog
retrieval, then multiple refinement iterations, then checkout — integrated
with the Shopify Storefront API for an end-to-end natural-language-to-purchase
flow. The key design decision was RAG-based grounded retrieval over real
product catalogs: the LLM's tool calls are constrained to in-stock items, which
eliminates hallucinated part recommendations and makes real-money transactions
safe. Built in Python.
