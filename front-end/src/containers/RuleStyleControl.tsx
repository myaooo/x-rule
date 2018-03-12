import * as React from 'react';
import { Action } from 'redux';
import { connect } from 'react-redux';
import { Slider, Row, Col, Radio } from 'antd';
import { RootState, Dispatch, changeRuleStyles, RuleStyles } from '../store';

const RadioButton = Radio.Button;
const RadioGroup = Radio.Group;

export interface RuleStyleStateProps {
  styles: RuleStyles;
}

export interface RuleStyleDispatchProps {
  changeStyles: (newStyles: Partial<RuleStyles>) => Action;
}

export interface RuleStyleControlProps extends RuleStyleStateProps, RuleStyleDispatchProps {
}

class RuleStyleControl extends React.Component<RuleStyleControlProps, any> {
  constructor(props: RuleStyleControlProps) {
    super(props);
  }
  // changeStyles(value: number) {
  //   this.props.changeStyles({width: value});
  // }
  render() {
    const {changeStyles} = this.props;
    return (
      <div style={{ paddingLeft: 12, fontSize: 12 }}>

         <Row>
          <Col span={10}>
            <span>Flow Width: </span>
          </Col>
          <Col span={14}>
            <Slider 
              min={5}
              max={100}
              value={this.props.styles.flowWidth}
              step={1}
              onChange={(flowWidth: number) => changeStyles({flowWidth})}
            />
          </Col>
        </Row>

        <Row>
          <Col span={10}>
            <span>Rect Width: </span>
          </Col>
          <Col span={14}>
            <Slider 
              min={5}
              max={100}
              value={this.props.styles.rectWidth}
              step={1}
              onChange={(rectWidth: number) => changeStyles({rectWidth})}
            />
          </Col>
        </Row>
        <Row>
          <Col span={10}>
            <span>Rect Height: </span>
          </Col>
          <Col span={14}>
            <Slider 
              min={5}
              max={100}
              value={this.props.styles.rectHeight}
              step={1}
              onChange={(rectHeight: number) => changeStyles({rectHeight})}
            />
          </Col>
        </Row>

        <Row style={{marginTop: 8}}>
          <Col span={10}>
            <span>Mode: </span>
          </Col>
          <Col span={14}>
            <RadioGroup 
              value={this.props.styles.mode}
              onChange={(e) => changeStyles({mode: e.target.value as 'matrix' | 'list'})}
              size="small"
            >
              <RadioButton value="list">List</RadioButton>
              <RadioButton value="matrix">Matrix</RadioButton>
            </RadioGroup>
          </Col>
        </Row>

        {/* </Slider> */}
      </div>
    );
  }
}

const mapStateToProps = (state: RootState): RuleStyleStateProps => {
  return {
    styles: state.ruleStyles,
  };
};

const mapDispatchToProps = (dispatch: Dispatch, ownProps: any): RuleStyleDispatchProps => {
  return {
    // loadModel: bindActionCreators(getModel, dispatch),
    changeStyles: (newStyles: Partial<RuleStyles>) => dispatch(changeRuleStyles(newStyles))
  };
};

export default connect(mapStateToProps, mapDispatchToProps)(RuleStyleControl);
