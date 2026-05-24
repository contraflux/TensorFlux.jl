import { Link, useLocation } from 'react-router-dom';
import { useState, useEffect } from 'react';
import style from './topbar.module.css';
import github from '../../assets/github.svg'

export default function Topbar() {
    return (
        <header className={style.topbar}>
            <Link className={style.logo} to="/home">
                <p>TensorFlux.jl</p>
            </Link>
            <nav className={style.navbar}>
                <Link className={style.navitem} to="/learn/getting-started">
                    <p>Learn</p>
                </Link>
                <Link className={style.navitem} to="/reference/geometric-objects">
                    <p>Reference</p>
                </Link>
                <Link className={style.navitem}>
                    <p>Examples</p>
                </Link>
            </nav>
            <div className={style.vbar}></div>
            <a className={style.external} href='https://github.com/contraflux/TensorFlux.jl' target='_blank'>
                <img src={github} height="25px"></img>
            </a>
        </header>
    );
}