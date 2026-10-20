#### Document- and field-level security

A role entry that grants `read` on a view name must not carry DLS or FLS restrictions. If it does, the query fails with a `403`. DLS/FLS on the underlying indices is applied normally. Refer to [view privileges](/reference/query-languages/esql/esql-views.md#esql-views-privileges).

#### Branching inside views

Commands that also generate branched query plans
(`FORK`, [subqueries](/reference/query-languages/esql/esql-subquery.md))
are usually not allowed within a view definition, unless the branch points
can be merged via
[query compaction](/reference/query-languages/esql/esql-views.md#query-compaction).

Views may be nested up to depth 10. Branching and nesting are allowed in
combination as long as there is never more than one branch point:

* A view can contain subqueries, but that view cannot be used together with
  other views, and the subqueries can only reference nested views that contain
  no further branching.
* A subquery can contain views, but those views must not introduce any
  additional branch points via subqueries or `FORK`.

#### Cross-cluster and serverless

Views can reference remote indices in [cross-cluster search](/reference/query-languages/esql/esql-cross-clusters.md), but you can't query a view that is defined on a remote cluster.

::::{applies-switch}
:::{applies-item} { "stack": "preview 9.6+", "serverless": "preview" }
Remote views are invisible to source resolution. An exact reference such as `FROM cluster:view` returns an `Unknown index` error, and wildcard expressions don't match remote views.
:::
:::{applies-item} stack: preview 9.4-9.5
A query fails if its source expression matches a remote view. A query also fails if a remote index matches a local view name.
:::
::::

Views are available in serverless and [cross-project search](/reference/query-languages/esql/esql-cross-serverless-projects.md), but with some limitations:
:::{include} ../common/cps_view_limitations.md
:::

#### Query DSL filters

::::{applies-switch}
:::{applies-item} { "stack": "preview 9.6+", "serverless": "preview" }
A Query DSL filter on the request applies to the output of a view, after the commands in the view definition. If Elasticsearch can't translate a filter clause to ES|QL, it omits that clause for the view and returns a warning. Check the response warnings because omitting a clause can produce more results than the complete filter would.
:::
:::{applies-item} stack: preview 9.4-9.5
A Query DSL filter on the request applies to the source indices in the view definition, not to the view output.
:::
::::

#### Query settings

View definitions handle `SET` differently across versions:

::::{applies-switch}
:::{applies-item} { "stack": "preview 9.6+", "serverless": "preview" }
A view definition can set `unmapped_fields`, `wildcards_match_views`, and `wildcards_match_datasets`. Each setting applies only to that view's body and doesn't enter a nested view. `unmapped_fields` doesn't support `LOAD_ALL`. Every other setting, including `time_zone`, is rejected in a view definition.
:::
:::{applies-item} stack: preview 9.4-9.5
A view definition can't contain a `SET` directive.
:::
::::

##### Settings from the calling query
```yaml {applies_to}
stack: preview 9.6+
serverless: preview
```

The calling query's values for `unmapped_fields`, `wildcards_match_views`, and `wildcards_match_datasets` stop at the view boundary. If the view definition doesn't set one of these values, its default applies inside the view body, regardless of the caller's value or the cluster default.

* The caller's `unmapped_fields` setting doesn't change field resolution inside the view body.
* The caller's `wildcards_match_views` and `wildcards_match_datasets` settings don't change wildcard resolution inside the view body. A wildcard in a view body matches views or datasets only when the view definition enables the corresponding setting. Exact names work regardless of these settings.

Other settings from the calling query that control the query execution context, such as `time_zone`, can apply inside a view.

#### Query parameters

Query parameters are not allowed in the view definition, and therefore query
parameters in the main query will never impact the view results.

#### Validation and dependencies

When you create or update a view, Elasticsearch validates its name, size, and query syntax. It doesn't check whether referenced indices, fields, datasets, or views exist, or whether nested views form a cycle. This allows you to create resources in any order. Elasticsearch reports dependency problems when you query the view.

Deleting a resource doesn't check whether a view depends on it. Deleting an index, data stream, alias, dataset, or nested view can therefore cause existing views to fail when queried.
