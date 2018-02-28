import { ModelBase } from './index';

// Tree Model

export interface LeafNode {
  readonly value: number[];
  readonly impurity: number;
  collapsed?: boolean;
  readonly idx: number;
  readonly output: number;
}

export interface InternalNode extends LeafNode {
  readonly left: TreeNode;
  readonly right: TreeNode;
  readonly feature: number;
  readonly threshold: number;
}

export type TreeNode = InternalNode | LeafNode;

export interface TreeModel extends ModelBase {
  readonly type: 'tree';
  readonly root: TreeNode;
  readonly nNodes: number;
  readonly maxDepth: number;
}

export function isLeafNode(node: TreeNode): node is LeafNode {
  return (<InternalNode> node).left === undefined;
}

export function isInternalNode(node: TreeNode): node is InternalNode {
  return (<InternalNode> node).left !== undefined;
}

export function isTreeModel(model: ModelBase): model is TreeModel {
  return model.type === 'tree';
}
