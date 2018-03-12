import * as d3 from 'd3';
import { ColorType, Painter, labelColor } from '../Painters';
import * as nt from '../../service/num';
import { Rule, Condition, Streams, DataSet } from '../../models';
import { RuleList } from '../../models/ruleModel';
import { defaultDuration } from '../Painters/Painter';
import RowPainter from './RowPainter';
import { RuleX, ConditionX } from './models';
import { ConditionalStreams, isConditionalStreams } from '../../models/data';

function computeExistingFeatures(rules: Rule[]) {
  const ruleFeatures = rules.slice(0, -1).map((r: Rule) => r.conditions.map((c: Condition) => c.feature));
  const features = Array.from(new Set(ruleFeatures.reduce((a, b) => a.concat(b)))).sort();
  return features;
}

interface OptionalParams {
  outputWidth: number;
  // height: number;
  duration: number;
  fontSize: number;
  paddingX: number;
  paddingY: number;
  transform: string;
  elemWidth: number;
  color: ColorType;
  expandFactor: [number, number];
}

export interface RuleMatrixParams extends Partial<OptionalParams> {
  dataset?: DataSet;
  streams?: Streams | ConditionalStreams;
  features: number[];
  feature2Idx: number[];
  activeFeatures?: Set<number>;
  // outputs?: number[];
  supports?: number[];
  xs: number[];
  widths: number[];
  onClick?: (feature: number) => void;
}

export default class RuleMatrixPainter implements Painter<RuleList, RuleMatrixParams> {
  public static defaultParams: OptionalParams = {
    // width: 200,
    // height: 50,
    color: labelColor,
    elemWidth: 30,
    duration: defaultDuration,
    fontSize: 12,
    paddingX: 0.1,
    paddingY: 0.2,
    // buttonSize: 12,
    outputWidth: 300,
    transform: '',
    expandFactor: [4, 3],
    // onClick: () => null
    // interval: 20,
  };
  private params: RuleMatrixParams & OptionalParams;
  private model: RuleList;
  private rules: RuleX[];
  private features: number[];
  private f2Idx: number[];
  private expandedElements: Set<string>;
  private rowPainter: RowPainter;
  // private selector: d3.Selection<SVGElement, any, any, any>;
  constructor() {
    // this.onClick = this.onClick.bind(this);
    this.expandedElements = new Set();
    this.rowPainter = new RowPainter();
  }
  public expandElement(rule: number, feature: number) {
    this.expandedElements.add(`${rule}-${feature}`);
  }
  public collapseElement(rule: number, feature: number) {
    this.expandedElements.delete(`${rule}-${feature}`);
  }
  feature2Idx(f: number) {
    // if (f >= this.f2Idx.length) return -1;
    return this.f2Idx[f];
  }
  public update(params: RuleMatrixParams) {
    this.params = { ...(RuleMatrixPainter.defaultParams), ...(this.params), ...params };
    return this;
  }
  public data(model: RuleList) {
    if (this.model !== model) {
      const features = computeExistingFeatures(model.rules);
      const f2Idx = new Array(model.nFeatures).fill(-1);
      features.forEach((f: number, i: number) => f2Idx[f] = i);
      this.features = features;
      this.f2Idx = f2Idx;
      this.rules = model.rules.map((r) => {
        const {conditions, ...rest} = r;
        const conditionXs: ConditionX[] = r.conditions.map((c) => ({
          ...c, title: '', x: 0, width: 0, interval: [null, null] as [null, null],
          expanded: false
        }));
        return {
          ...rest, conditions: conditionXs, height: 0, x: 0, y: 0, width: 0,
        };
      });
    }
    this.model = model;
    return this;
  }
  public updatePos() {
    const {expandFactor, elemWidth, paddingX, paddingY, dataset, streams} = this.params;
   
    // compute active sets
    const activeRules = new Set<number>();
    const activeFeatures = new Set<number>();
    this.expandedElements.forEach((s: string) => {
      const rf = s.split('-');
      activeRules.add(Number(rf[0]));
      activeFeatures.add(Number(rf[1]));
    });

    // compute the widths and heights
    const expandWidth = elemWidth * expandFactor[0];
    const expandHeight = elemWidth * expandFactor[1];

    const featureWidths = 
      this.features.map((f: number) => (activeFeatures.has(f) ? expandWidth : elemWidth));
    const ruleHeights = 
      this.rules.map((r: Rule, i: number) => (activeRules.has(i) ? expandHeight : elemWidth));

    let ys = ruleHeights.map((h) => h + paddingY);
    ys = [0, ...(nt.cumsum(ys.slice(0, -1)))];

    let xs = featureWidths.map((w: number) => w + paddingX * elemWidth);
    xs = [0, ...(nt.cumsum(xs.slice(0, -1)))];

    const width = xs[xs.length - 1] + featureWidths[featureWidths.length - 1];

    // update ruleX positions
    this.rules.forEach((r, i) => {
      r.y = ys[i]; 
      r.height = ruleHeights[i];
      r.width = width;
    });
    // update conditionX positions
    this.rules.forEach((r, i) => {
      r.conditions.forEach((c) => {
        c.x = xs[c.feature];
        c.width = featureWidths[c.feature];
        if (dataset) {
          c.title = dataset.categoryDescription(c.feature, c.category);
          c.interval = dataset.categoryInterval(c.feature, c.category);
        }
        if (streams) {
          if (isConditionalStreams(streams)) c.stream = streams[i][c.feature];
          else c.stream = streams[c.feature];
        }
      });
    });

    return;
  }
  public render(selector: d3.Selection<SVGElement, any, any, any>) {
    const { duration } = this.params;
    this.updatePos();
    // Root Group
    selector
      .selectAll('g.rules')
      .data(['flows'])
      .enter()
      .append('g')
      .attr('class', 'rules');
    
    // Joined
    const rule = selector
      .select('g.rules')
      .selectAll<SVGGElement, RuleX>('g.rule')
      .data(this.rules);

    // Enter
    const ruleEnter = rule.enter().append<SVGGElement>('g').attr('class', 'rule');

    // Update
    const ruleUpdate = ruleEnter.merge(rule);
    ruleUpdate
      .transition()
      .duration(duration)
      .attr('transform', (d: RuleX) => `translate(0, ${d.y})`);
    
    this.rowPainter.update({
      feature2Idx: this.feature2Idx,
    });
    ruleUpdate.each((d: RuleX, i: number, nodes) => {
      this.rowPainter.data(d).render(d3.select(nodes[i]));
    });

    // Exit
    rule.exit()
      .transition()
      .duration(duration)
      .attr('transform', `translate(-500, 0)`)
      .attr('fill-opacity', 1e-6)
      .remove();

    return this;
  }
}
