import * as d3 from 'd3';
import * as nt from '../../service/num';
import { Painter, ColorType, labelColor, defaultDuration } from '../Painters';
import { RuleX } from './models';
import { registerStripePattern } from '../../service/utils';
import { isRuleGroup } from '../../models/ruleModel';

interface OptionalSupportParams {
  duration: number;
  color: ColorType;
}

interface SupportParams extends Partial<OptionalSupportParams> {
  widthFactor: number;
  height: number;
}

type SupportData = number[] | number[][];

function isMat(a: number[] | number[][]): a is number[][] {
  return Array.isArray(a[0]);
}

function registerPatterns(color: ColorType, keys: number[]) {
  return keys.map((key) => registerStripePattern(color(key), 2, 3));
}

export class SupportPainter implements Painter<SupportData, SupportParams> {
  public static defaultParams: OptionalSupportParams = {
    color: labelColor,
    duration: defaultDuration,
  };
  private params: SupportParams & OptionalSupportParams;
  private support: SupportData;
  update(params: SupportParams): this {
    this.params = {...(SupportPainter.defaultParams), ...(this.params), ...params};
    return this;
  }
  data(newData: SupportData): this {
    this.support = newData;
    return this;
  }
  render<GElement extends d3.BaseType>(
    selector: d3.Selection<SVGGElement, any, GElement, any>,
  ): this {
    const support = this.support;
    if (isMat(support)) {
      this.renderMat(selector, support);
    } else {
      this.renderMat(selector, []);
      this.renderSimple(selector, support);
    }
    return this;
  }
  renderSimple<GElement extends d3.BaseType>(
    selector: d3.Selection<SVGGElement, any, GElement, any>,
    support: number[]
  ): this {
    const {duration, height, widthFactor, color} = this.params;
    
    const xs = [0, ...(nt.cumsum(support))];

    // Render
    // Join
    const rects = selector.selectAll('rect.mo-support').data(support);
    // Enter
    const rectsEnter = rects.enter().append('rect').attr('class', 'mo-support')
      .attr('height', height);
    // Update
    const rectsUpdate = rectsEnter.merge(rects)
      .style('fill', (d, i) => color(i));
    // Transition
    rectsUpdate.transition().duration(duration)
      .attr('width', (d) => d * widthFactor)
      .attr('x', (d, i) => xs[i] * widthFactor + i * 1.5)
      .attr('height', height);
    // Exit
    rects.exit().transition().duration(duration)
      .attr('width', 1e-6).remove();
    return this;
  }

  renderMat<GElement extends d3.BaseType>(
    selector: d3.Selection<SVGGElement, any, GElement, any>,
    support: number[][]
  ): this {
    const { height, widthFactor, duration, color } = this.params;
    const trueLabels = support.length ? nt.sumVec(support) : [];
    // const total = nt.sum(trueLabels);
    // const width = total * widthFactor;
    const widths = trueLabels.map((l) => l * widthFactor);
    const xs = [0, ...(nt.cumsum(widths))];
    const ys = support.map((s, i) => s[i] / trueLabels[i] * height);
    // const heights = ys.map((y) => height - y);

    // Render True Rects
    const trueData = d3.transpose<number>(support)
      .map((s, i) => ({width: widths[i], x: xs[i], height: ys[i], data: s, label: i}))
      .filter(v => v.width > 0);
    // Join
    const rects = selector.selectAll('rect.mo-support')
      .data(trueData);
    // Enter
    const rectsEnter = rects.enter().append('rect').attr('class', 'mo-support')
      .attr('height', height);
    // Update
    const rectsUpdate = rectsEnter.merge(rects)
      .style('fill', d => color(d.label))
      .style('stroke', d => color(d.label));
    // Transition
    rectsUpdate.transition().duration(duration)
      .attr('width', d => d.width)
      .attr('x', (d, i) => d.x + i * 1.5)
      .attr('height', d => d.height);
    // Exit
    rects.exit().transition().duration(duration)
      .attr('width', 1e-6).remove();

    // Register the stripes
    const stripeNames = registerPatterns(color, d3.range(trueLabels.length));
    
    // Render the misclassified part using stripes
    const root = selector.selectAll<SVGGElement, number[]>('g.mo-support-mat')
      .data(trueData);
    // enter
    const rootEnter = root.enter().append<SVGGElement>('g')
      .attr('class', 'mo-support-mat');

    // update
    const rootUpdate = rootEnter.merge(root).style('stroke', d => color(d.label));

    // update transition
    rootUpdate.transition().duration(duration)
      .attr('transform', (d, i) => `translate(${d.x + i * 1.5},${d.height})`);

    // root exit
    const exitTransition = root.exit().transition().duration(duration).remove();
    exitTransition.selectAll('rect.mo-support-mat').attr('width', 1e-6).attr('x', 1e-6);

    // stripe rects
    const stripeRects = rootUpdate.selectAll('rect.mo-support-mat')
    .data((d) => {
      // const xs = [0, ...(nt.cumsum(d))];
      const base = nt.sum(d.data) - d.data[d.label];
      let factor = base ? d.width / base : 0;
      const _widths = d.data.map((v, j) => j === d.label ? 0 : v * factor);
      const _xs = [0, ...nt.cumsum(_widths)];
      // console.log(factor); // tslint:disable-line
      const ret = d.data.map((v, j) => ({
        height: height - d.height, 
        width: _widths[j], x: _xs[j], label: j
      }));
      return ret.filter(r => r.width > 0);
    });
    const stripeEnter = stripeRects.enter().append('rect')
      .attr('class', 'mo-support-mat').attr('height', d => d.height);
    const stripeUpdate = stripeEnter.merge(stripeEnter)
      // .classed('striped', d => d.striped)
      // .style('stroke', d => color(d.label))
      // .style('display', d => d.striped ? 'inline' : 'none')
      .style('fill', d => `url(#${stripeNames[d.label]})`);

    stripeUpdate.transition().duration(duration)
      .attr('height', d => d.height)
      .attr('width', d => d.width).attr('x', d => d.x);
    
    stripeRects.exit().transition().duration(duration)
      .attr('width', 1e-6).attr('x', 1e-6).remove();

    return this;
  }
}

interface OptionalParams {
  color: ColorType;
  duration: number;
  fontSize: number;
  widthFactor: number;
}

export interface OutputParams extends Partial<OptionalParams> {
  // feature2Idx: (feature: number) => number;
  onClick?: (feature: number, condition: number) => void;
}

export default class OutputPainter implements Painter<RuleX[], OutputParams> {
  public static defaultParams: OptionalParams = {
    color: labelColor,
    duration: defaultDuration,
    fontSize: 14,
    widthFactor: 200,
    // expandFactor: [4, 3],
  };
  private rules: RuleX[];
  private params: OutputParams & OptionalParams;
  private supportPainter: SupportPainter;
  constructor() {
    this.params = {...(OutputPainter.defaultParams)};
    this.supportPainter = new SupportPainter();
  }
  
  update(params: OutputParams): this {
    this.params = {...(OutputPainter.defaultParams), ...(this.params), ...params};
    return this;
  }
  data(newData: RuleX[]): this {
    this.rules = newData;
    return this;
  }
  render<GElement extends d3.BaseType>(
    selector: d3.Selection<SVGGElement, any, GElement, any>,
  ): this {
    const { duration } = this.params;
    const rules = this.rules;
    const collapseYs = new Map<string, number>();
    rules.forEach((r) => isRuleGroup(r) && r.rules.forEach((_r) => collapseYs.set(`o-${_r.idx}`, r.y)));
    this.renderHeader(selector);
    // ROOT Group
    const groups = selector.selectAll<SVGGElement, {}>('g.matrix-outputs')
      .data<RuleX>(rules, function (r: RuleX) { return r ? `o-${r.idx}` : this.id; });
    // Enter
    const groupsEnter = groups.enter()
      .append<SVGGElement>('g')
      .attr('class', 'matrix-outputs')
      .attr('id', d => `o-${d.idx}`);
    // Update
    const groupsUpdate = groupsEnter.merge(groups)
      .classed('hidden', false).classed('visible', true);
    const updateTransition = groupsUpdate.transition().duration(duration)
      .attr('transform', d => `translate(10,${d.y})`);

    this.renderOutputs(groupsEnter, groupsUpdate, updateTransition);
    this.renderSupports(groupsEnter, groupsUpdate);

    // Exit
    groups.exit()
      .classed('hidden', true).classed('visible', false)
      .transition().duration(duration)
      .attr('transform', (d, i, nodes) => 
        `translate(5,${collapseYs.get(nodes[i].id)})`);
    return this;
  }

  public renderHeader(root: d3.Selection<SVGGElement, any, d3.BaseType, any>): this {
    // make sure the group exists
    // console.log('here'); // tslint:disable-line
    const {duration} = this.params;
    const rules = this.rules;
    const totalSupport = nt.sum(rules.map((r) => r.totalSupport));
    const confidence = nt.sum(rules.map((r) => r.totalSupport * r.output[r.label])) / totalSupport;
    root.selectAll('g.mo-headers').data(['g']).enter()
      .append('g').attr('class', 'mo-headers').attr('transform', 'translate(0,-20)');
    
    const headerTexts = [`Confidence (${confidence.toFixed(2)})`, 'Support'];
    const headerXs = [12, 70];
    const rectWidths = [120, 60];
    const fillRatios = [confidence, 0];
    const headers = root.select('g.mo-headers');

    const header = headers.selectAll('g.mo-header').data(headerTexts);
    const headerEnter = header.enter().append('g').attr('class', 'mo-header')
      .attr('transform', (d, i) => `translate(${headerXs[i]},0) rotate(-50)`)
      .style('font-size', 14);

    // rects
    headerEnter.append('rect')
      .style('stroke-width', 1).style('stroke', '#1890ff').style('fill', '#fff')
      .attr('width', (d, i) => rectWidths[i]).attr('height', 20)
      .attr('rx', 2).attr('ry', 2);

    // rects
    headerEnter.append('rect').attr('class', 'mo-header-fill')
      .style('stroke-width', 1).style('stroke', '#1890ff')
      .style('fill', '#1890ff').style('fill-opacity', 0.1)
      .attr('height', 20)
      .attr('rx', 2).attr('ry', 2);

    // texts
    headerEnter.append('text')
      .attr('text-anchor', 'start')
      .attr('fill', '#1890ff')
      .attr('dx', 3).attr('dy', 15);
    // Update
    const headerUpdate = headerEnter.merge(header);
    headerUpdate.select('text').text(d => d);
    headerUpdate.transition().duration(duration).select('rect.mo-header-fill')
      .attr('width', (d, i) => fillRatios[i] * rectWidths[i]);

    // textsEnter.merge(texts).text(d => d);
    return this;
  }

  public renderOutputs(
    enter: d3.Selection<SVGGElement, RuleX, SVGGElement, RuleX[]>,
    update: d3.Selection<SVGGElement, RuleX, SVGGElement, RuleX[]>,
    updateTransition: d3.Transition<SVGGElement, RuleX, SVGGElement, RuleX[]>
  ): this {
    const {fontSize, color, duration} = this.params;
    const outputWidth = fontSize * 2;

    // *Output Texts*
    // Enter
    enter.append('text').attr('text-anchor', 'start');
    // Update
    update.select('text')
      .attr('font-size', d => isRuleGroup(d) ? fontSize * 0.8 : fontSize)
      .attr('dy', d => isRuleGroup(d) ? fontSize * 0.8 : fontSize)
      .text((d: RuleX) =>
        isRuleGroup(d) ? '' : (Math.round(d.output[d.label] * 100) / 100).toFixed(2)
      );  // confidence as text
    // Transition
    updateTransition.select('text')
      .style('fill', d => 
        d3.interpolateRgb.gamma(2.2)('#ddd', color(d.label))(d.output[d.label] * 2 - 1)
      );

    // *Output Bars*
    // Group
    enter.append('g').attr('class', 'mo-outputs');
    // Join
    const rects = update.select('g.mo-outputs')
      .attr('transform', d => `translate(0, ${fontSize * (isRuleGroup(d) ? 0.8 : 1) + 1})`) // update pos
      .selectAll('rect')
      .data((d: RuleX) => {
        if (isRuleGroup(d)) return [];
        const widths = d.output.map((o) => o * outputWidth);
        const xs = [0, ...(nt.cumsum(widths.slice(0, -1)))];
        return d.output.map((o: number, i: number) => ({
          width: widths[i],
          x: xs[i] + i,
        }));
      });
    // Enter + Merge
    const rectsUpdate = rects.enter()
      .append('rect').attr('x', 0).attr('width', 0).attr('height', 3)
      .merge(rects);

    // transition
    rectsUpdate.transition().duration(duration)
      .attr('x', d => d.x).attr('width', d => d.width)
      .style('fill', (d, i) => color(i));
    
    rects.exit().transition().duration(duration)
      .attr('x', 0).attr('width', 0).remove();
    return this;
  }

  renderSupports(
    enter: d3.Selection<SVGGElement, RuleX, SVGGElement, RuleX[]>,
    update: d3.Selection<SVGGElement, RuleX, SVGGElement, RuleX[]>
  ): this {
    const { duration, fontSize, widthFactor } = this.params;
    // Enter
    enter.append('g').attr('class', 'mo-supports');
    // Update
    const supports = update.select<SVGGElement>('g.mo-supports');
    supports.transition().duration(duration)
      .attr('transform', `translate(${fontSize * 3},0)`);

    // supports
    supports.each(({support, height}, i, nodes) => 
      support && this.supportPainter
        .update({widthFactor, height})
        .data(support)
        .render(d3.select(nodes[i]))
    );

    return this;
  }
}
