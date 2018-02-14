// export function max

import {
  isTreeModel,
  isRuleModel,
  ModelBase,
  Rule,
  Condition,
  TreeNode,
  isInternalNode,
  InternalNode
} from '../models';

export function countFeatureFreq(model: ModelBase, nFeatures: number): (number | undefined)[] {

  const counts = new Array(nFeatures);
  counts.fill(0);
  if (isRuleModel(model)) {
    model.rules.forEach((rule: Rule) => {
      rule.conditions.forEach((c: Condition) => {
        counts[c.feature]++;
      });
    });
  } else if (isTreeModel(model)) {
    traverseTree(model.root, (node: TreeNode) => {
      if (isInternalNode(node)) {
        counts[node.feature]++;
      }
    });
  }

  return counts;
}

export function hasLeftChild(node: TreeNode) {
  return isInternalNode(node) && node.left;
}

export function hasRightChild(node: TreeNode) {
  return isInternalNode(node) && node.right;
}

// Performs pre-order traversal on a tree
export function traverseTree(source: TreeNode, fn: (node: TreeNode, i: number) => void) {
  let idx = 0;
  const _traverse = (node: TreeNode) => {
    // root
    fn(node, idx++);
    // left
    if (hasLeftChild(node)) {
      _traverse((node as InternalNode).left);
    }
    // right
    if (hasRightChild(node)) {
      _traverse((node as InternalNode).right);
    }
  };
  _traverse(source);
}

const MAX_STR_LEN = 16;
const CUT_SIZE = (MAX_STR_LEN - 2) / 2;

export function condition2String(
  featureName: string, 
  category: (number | null)[] | number
): { tspan: string; title: string } {
  const abrString = featureName.length > MAX_STR_LEN
    ? `"${featureName.substr(0, CUT_SIZE)}…${featureName.substr(-CUT_SIZE, CUT_SIZE)}"`
    : featureName;
  let featureMap = (feature: string): string => `${feature} is any`;
  if (typeof category === 'number') {
    featureMap = (feature: string) => `${feature} = ${category}`;
  } else {
    const low = category[0];
    const high = category[1];
    if (low === null && high === null) featureMap = (feature: string) => `${feature} is any`;
    else {
      const lowString = low !== null ? `${low.toPrecision(3)} < ` : '';
      const highString = high !== null ? ` < ${high.toPrecision(3)}` : '';
      featureMap = (feature: string) => lowString + feature + highString;
    }
  }
  return {
    tspan: featureMap(abrString),
    title: featureMap(featureName)
  };
}

export function sum(arr: Array<number>) {
  let _sum = 0;
  for (let e of arr) {
    _sum += e;
  }
  return _sum;
}

export function collapseInit(root: TreeNode, threshold: number = 0.2) {
  console.log('init collpase attr!'); // tslint:disable-line
  const totalSupport = sum(root.value);
  const minDisplaySupport = Math.floor(threshold * totalSupport);
  traverseTree(root, (node: TreeNode) => {
    node.collapsed = sum(node.value) < minDisplaySupport;
  });
}