import * as React from 'react';
import { connect } from 'react-redux';
import { Row, Col, Switch } from 'antd';
import { RootState, Dispatch, changeSettingsAndFetchData, Settings } from '../store';

export interface SettingsStateProps {
  settings: Settings;
}

export interface SettingsDispatchProps {
  updateSettings: (newSettings: Partial<Settings>) => void;
}

export interface SettingsControlProps extends SettingsStateProps, SettingsDispatchProps {
}

class SettingsControl extends React.Component<SettingsControlProps, any> {
  constructor(props: SettingsControlProps) {
    super(props);
  }
  // changeStyles(value: number) {
  //   this.props.changeStyles({width: value});
  // }
  render() {
    const {updateSettings} = this.props;
    return (
      <div style={{ paddingLeft: 12 }}>

        <Row style={{marginTop: 8}}>
          <Col span={10}>
            <span>Conditional: </span>
          </Col>
          <Col span={14}>
            <Switch 
              checked={this.props.settings.conditional}
              onChange={(conditional) => updateSettings({conditional})}
              size="small"
            />
          </Col>
        </Row>
        {/* </Slider> */}
      </div>
    );
  }
}

const mapStateToProps = (state: RootState): SettingsStateProps => {
  return {
    settings: state.settings,
  };
};

const mapDispatchToProps = (dispatch: Dispatch, ownProps: any): SettingsDispatchProps => {
  return {
    // loadModel: bindActionCreators(getModel, dispatch),
    updateSettings: (newSettings: Partial<Settings>) => dispatch(changeSettingsAndFetchData(newSettings))
  };
};

export default connect(mapStateToProps, mapDispatchToProps)(SettingsControl);
